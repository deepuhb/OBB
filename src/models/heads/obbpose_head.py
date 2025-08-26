
# src/models/heads/obbpose_head.py
from __future__ import annotations
from typing import List, Tuple

import math
import torch
import torch.nn as nn

class OBBPoseHead(nn.Module):
    """
    Multi-scale head for OBB detection and optional keypoint hints.

    Detection channel layout per level (C = 7 + nc):
        [ tx, ty, tw, th, sinθ, cosθ, obj, (cls0..cls{nc-1}) ]

    Notes:
    - tx, ty are raw offsets (sigmoid applied in decoder).
    - tw, th are log-width/height (exp applied in decoder).
    - sinθ, cosθ come from linear conv; we L2-normalize in forward (non-inplace).
    - obj and cls are raw logits.
    """

    def __init__(self, ch: Tuple[int, int, int], num_classes: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        c3, c4, c5 = [int(x) for x in ch]

        det_out = 7 + self.num_classes  # [tx,ty,tw,th,sin,cos,obj,(cls...)]
        kpt_out = 3  # placeholder for optional kpt maps

        self.det3 = nn.Sequential(
            nn.Conv2d(c3, c3, 3, 1, 1), nn.BatchNorm2d(c3), nn.SiLU(inplace=True),
            nn.Conv2d(c3, det_out, 1, 1, 0)
        )
        self.det4 = nn.Sequential(
            nn.Conv2d(c4, c4, 3, 1, 1), nn.BatchNorm2d(c4), nn.SiLU(inplace=True),
            nn.Conv2d(c4, det_out, 1, 1, 0)
        )
        self.det5 = nn.Sequential(
            nn.Conv2d(c5, c5, 3, 1, 1), nn.BatchNorm2d(c5), nn.SiLU(inplace=True),
            nn.Conv2d(c5, det_out, 1, 1, 0)
        )

        self.kp3 = nn.Sequential(
            nn.Conv2d(c3, c3, 3, 1, 1), nn.BatchNorm2d(c3), nn.SiLU(inplace=True),
            nn.Conv2d(c3, kpt_out, 1, 1, 0)
        )
        self.kp4 = nn.Sequential(
            nn.Conv2d(c4, c4, 3, 1, 1), nn.BatchNorm2d(c4), nn.SiLU(inplace=True),
            nn.Conv2d(c4, kpt_out, 1, 1, 0)
        )
        self.kp5 = nn.Sequential(
            nn.Conv2d(c5, c5, 3, 1, 1), nn.BatchNorm2d(c5), nn.SiLU(inplace=True),
            nn.Conv2d(c5, kpt_out, 1, 1, 0)
        )

        self._assert_once = False

    def forward(self, feats: List[torch.Tensor]):
        """
        Args:
            feats: [P3, P4, P5] feature maps
        Returns:
            det:  list of 3 tensors (B, 7+nc, H_l, W_l)
            kpm:  list of 3 tensors (B, 3,     H_l, W_l)
        """
        assert isinstance(feats, (list, tuple)) and len(feats) == 3, \
            "OBBPoseHead.forward expects 3 feature maps [P3,P4,P5]"

        n3, d4, d5 = feats
        det = [self.det3(n3), self.det4(d4), self.det5(d5)]

        # One-time channel check
        if not self._assert_once:
            for i, lvl in enumerate(("P3", "P4", "P5")):
                C = int(det[i].shape[1])
                expected = 7 + self.num_classes
                assert C == expected, \
                    f"OBBPoseHead({lvl}) channels={C}, expected {expected}=[tx,ty,tw,th,sin,cos,obj,(cls...)]"
            self._assert_once = True

        # Non-inplace normalization of (sin,cos)
        new_det = []
        for i in range(3):
            d = det[i]
            if d.shape[1] >= 7:
                ang = d[:, 4:6, ...]
                norm = ang.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
                ang_norm = ang / norm
                d = torch.cat([d[:, :4, ...], ang_norm, d[:, 6:, ...]], dim=1).contiguous()
            new_det.append(d)
        det = new_det

        kpm = [self.kp3(n3), self.kp4(d4), self.kp5(d5)]
        return det, kpm

    def decode_obb_from_pyramids(
            self,
            det_maps,
            imgs,
            *,
            strides=(8, 16, 32),
            conf_thres=None,  # old name
            score_thr=None,  # evaluator's name
            max_det=300,
            multi_label=False,
            agnostic=False,
            use_nms=False,  # evaluator handles NMS; keep False here
            **kwargs,
    ):
        """
        Returns per-image dicts:
          {"boxes": (N,5) [cx,cy,w,h,theta(rad)], "scores": (N,), "labels": (N,)}
        Channels per level are [tx,ty,tw,th,sin,cos,obj,(cls...)]  (7 + nc)
        Notes:
          - Robust fp32 decode (safe exp, angle normalisation, LE-90 canonicalisation).
          - Accepts both 'conf_thres' and 'score_thr' for compatibility.
        """
        import math
        import torch

        # pick threshold compatibly
        thr = conf_thres if conf_thres is not None else score_thr
        if thr is None:
            thr = 0.25  # sensible default

        B, _, H, W = imgs.shape
        img_h, img_w = int(H), int(W)

        def _safe_exp(x, lo=-6.0, hi=6.0):
            # clamp logits to avoid fp16 overflow; exp(6) ≈ 403
            return torch.exp(x.clamp(min=lo, max=hi))

        def _le90(w, h, ang):
            # tensor-safe LE-90: ensure w >= h; rotate θ by +π/2 where swapped; wrap to [-π/2, π/2)
            swap = (w < h)  # torch.bool mask
            w2 = torch.where(swap, h, w)
            h2 = torch.where(swap, w, h)
            ang2 = torch.where(swap, ang + math.pi / 2.0, ang)
            ang2 = torch.remainder(ang2 + math.pi / 2.0, math.pi) - math.pi / 2.0
            return w2, h2, ang2

        outs_per_level = []
        # Do all math in fp32 for numerical safety (even under AMP)
        with torch.cuda.amp.autocast(enabled=False):
            for li, (m, s) in enumerate(zip(det_maps, strides)):
                # m: [B, C, h, w], C=(7+nc)
                p = m.float().contiguous()
                b, c, h, w = p.shape
                nc = c - 7
                if b != B:
                    raise RuntimeError(f"Batch mismatch at level {li}: det_maps has B={b}, imgs has B={B}")

                # one-time debug
                if not hasattr(self, "_DBG_DECODE_ONCE"):
                    print(f"[DECODE ASSERT] level#{li} fmap={h}x{w}  auto_stride={s}")
                    if li == len(det_maps) - 1:
                        self._DBG_DECODE_ONCE = True

                # grid (fp32)
                gy, gx = torch.meshgrid(
                    torch.arange(h, device=p.device, dtype=torch.float32),
                    torch.arange(w, device=p.device, dtype=torch.float32),
                    indexing="ij",
                )
                gx = gx.view(1, 1, h, w).expand(b, -1, -1, -1)  # [B,1,h,w]
                gy = gy.view(1, 1, h, w).expand(b, -1, -1, -1)  # [B,1,h,w]

                # split channels
                tx = p[:, 0:1]  # [B,1,h,w]
                ty = p[:, 1:2]
                tw = p[:, 2:3]
                th = p[:, 3:4]
                sn = p[:, 4:5]
                cs = p[:, 5:6]
                obj = p[:, 6:7]
                cls = p[:, 7:] if nc > 0 else None  # [B,nc,h,w] or None

                # normalise (sin,cos) to unit circle
                denom = torch.sqrt(sn * sn + cs * cs + 1e-9)
                sn = sn / denom
                cs = cs / denom
                ang = torch.atan2(sn, cs)  # [B,1,h,w], radians

                # YOLO-style decode
                cx = (gx + torch.sigmoid(tx)) * float(s)
                cy = (gy + torch.sigmoid(ty)) * float(s)
                pw = _safe_exp(tw) * float(s)
                ph = _safe_exp(th) * float(s)

                # clamp sizes to avoid pathological IoU/NMS behavior
                max_wh = float(2.0 * max(img_h, img_w))
                pw = pw.clamp_(min=1.0, max=max_wh)
                ph = ph.clamp_(min=1.0, max=max_wh)

                # LE-90 canonicalisation
                pw, ph, ang = _le90(pw, ph, ang)

                # scores & labels
                obj_prob = torch.sigmoid(obj)  # [B,1,h,w]
                if cls is not None and cls.numel():
                    cls_prob = torch.sigmoid(cls)  # [B,nc,h,w]
                    cls_score, cls_idx = cls_prob.max(dim=1, keepdim=True)  # [B,1,h,w]
                    score = obj_prob * cls_score
                    labels = cls_idx.to(torch.long)  # [B,1,h,w]
                else:
                    score = obj_prob
                    labels = torch.zeros_like(obj, dtype=torch.long)  # [B,1,h,w]

                # threshold
                keep = score > float(thr)  # [B,1,h,w]
                if not keep.any():
                    outs_per_level.append([
                        {
                            "boxes": torch.zeros((0, 5), device=p.device, dtype=torch.float32),
                            "scores": torch.zeros((0,), device=p.device, dtype=torch.float32),
                            "labels": torch.zeros((0,), device=p.device, dtype=torch.long),
                        }
                        for _ in range(B)
                    ])
                    continue

                # flatten selected
                cx = cx[keep].view(-1)
                cy = cy[keep].view(-1)
                pw = pw[keep].view(-1)
                ph = ph[keep].view(-1)
                ang = ang[keep].view(-1)
                sc = score[keep].view(-1)
                lb = labels[keep].view(-1)

                # split by batch
                by, _, jy, ix = keep.nonzero(as_tuple=True)
                lvl_outs = []
                for bi in range(B):
                    sel = (by == bi)
                    if sel.any():
                        boxes = torch.stack([cx[sel], cy[sel], pw[sel], ph[sel], ang[sel]], dim=1)
                        scores = sc[sel]
                        labs = lb[sel]
                        if boxes.shape[0] > int(max_det):
                            topk = torch.topk(scores, k=int(max_det), dim=0).indices
                            boxes, scores, labs = boxes[topk], scores[topk], labs[topk]
                    else:
                        boxes = torch.zeros((0, 5), device=p.device, dtype=torch.float32)
                        scores = torch.zeros((0,), device=p.device, dtype=torch.float32)
                        labs = torch.zeros((0,), device=p.device, dtype=torch.long)
                    lvl_outs.append({"boxes": boxes, "scores": scores, "labels": labs})
                outs_per_level.append(lvl_outs)

        # fuse levels per image
        fused = []
        for bi in range(B):
            bi_boxes, bi_scores, bi_labels = [], [], []
            for lvl_outs in outs_per_level:
                bi_boxes.append(lvl_outs[bi]["boxes"])
                bi_scores.append(lvl_outs[bi]["scores"])
                bi_labels.append(lvl_outs[bi]["labels"])
            if bi_boxes:
                boxes = torch.cat(bi_boxes, dim=0)
                scores = torch.cat(bi_scores, dim=0)
                labels = torch.cat(bi_labels, dim=0)
                if boxes.shape[0] > int(max_det):
                    topk = torch.topk(scores, k=int(max_det), dim=0).indices
                    boxes, scores, labels = boxes[topk], scores[topk], labels[topk]
            else:
                boxes = torch.zeros((0, 5), device=imgs.device, dtype=torch.float32)
                scores = torch.zeros((0,), device=imgs.device, dtype=torch.float32)
                labels = torch.zeros((0,), device=imgs.device, dtype=torch.long)
            fused.append({"boxes": boxes, "scores": scores, "labels": labels})
        return fused
