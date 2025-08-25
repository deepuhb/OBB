
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


    @torch.no_grad()
    def decode_obb_from_pyramids(
            self,
            det_maps,
            imgs,
            *,
            conf_thres: float = 0.25,
            iou_thres: float = 0.45,
            max_det: int = 300,
            use_nms: bool = True,
            multi_label: bool = False,
            agnostic: bool = False,  # kept for signature compat
            strides=(8, 16, 32),  # ignored (we auto-infer stride)
    ):
        """
        det_maps: list of 3 tensors [B,C,H,W] with channels [tx,ty,tw,th,sin,cos,obj,(cls...)]
        imgs:     input batch [B,3,Himg,Wimg]
        Returns:  list of length B with dict(boxes [N,5: cx,cy,w,h,θ], scores [N], labels [N])
        """
        device = imgs.device
        dtype = imgs.dtype
        B, _, Himg, Wimg = imgs.shape
        assert isinstance(det_maps, (list, tuple)) and len(det_maps) == 3, "expected 3 FPN levels"

        def make_grid(h, w, dev):
            # [1,1,H,W] integer grid
            yv, xv = torch.meshgrid(
                torch.arange(h, device=dev),
                torch.arange(w, device=dev),
                indexing="ij",
            )
            return xv.view(1, 1, h, w), yv.view(1, 1, h, w)

        def aabb_from_obb(xywhθ):
            # xywhθ: [N,5] -> [N,4] (x1,y1,x2,y2) (axis-aligned)
            cx, cy, w, h = xywhθ[:, 0], xywhθ[:, 1], xywhθ[:, 2], xywhθ[:, 3]
            return torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dim=1)

        def nms_aabb(xyxy, scores, iou_thr):
            # Pure-torch NMS on axis-aligned boxes
            if xyxy.numel() == 0:
                return torch.empty((0,), dtype=torch.long, device=xyxy.device)
            x1, y1, x2, y2 = xyxy.unbind(1)
            areas = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
            order = torch.argsort(scores, descending=True)
            keep = []
            while order.numel():
                i = order[0]
                keep.append(i.item())
                if order.numel() == 1:
                    break
                xx1 = torch.maximum(x1[i], x1[order[1:]])
                yy1 = torch.maximum(y1[i], y1[order[1:]])
                xx2 = torch.minimum(x2[i], x2[order[1:]])
                yy2 = torch.minimum(y2[i], y2[order[1:]])
                w = (xx2 - xx1).clamp_min(0)
                h = (yy2 - yy1).clamp_min(0)
                inter = w * h
                union = areas[i] + areas[order[1:]] - inter + 1e-9
                iou = inter / union
                order = order[1:][iou <= iou_thr]
            return torch.as_tensor(keep, device=xyxy.device, dtype=torch.long)

        # one-time assert print
        if not hasattr(self, "_decode_once"):
            self._decode_once = False

        # per-image accumulators
        out_boxes = [[] for _ in range(B)]
        out_scores = [[] for _ in range(B)]
        out_labels = [[] for _ in range(B)]

        L = len(det_maps)
        max_det_per_level = max(1, math.ceil(max_det / L))

        for level_idx, dm in enumerate(det_maps):
            assert dm.dim() == 4, "det map must be (B,C,H,W)"
            Bb, C, H, W = dm.shape
            assert Bb == B
            nc = C - 7
            assert C == 7 + nc, f"Decoder got C={C}, expected 7+nc (got nc={nc})"

            # ---- AUTO-STRIDE (robust to order) ----
            s_h = int(round(Himg / H)) if H > 0 else 1
            s_w = int(round(Wimg / W)) if W > 0 else 1
            s = s_w if s_w == s_h else int(round((Himg / max(1, H) + Wimg / max(1, W)) * 0.5))
            if not self._decode_once:
                print(f"[DECODE ASSERT] level#{level_idx} fmap={H}x{W}  auto_stride={s}")

            # slices
            tx = dm[:, 0:1]  # [B,1,H,W]
            ty = dm[:, 1:2]
            tw = dm[:, 2:3]
            th = dm[:, 3:4]
            tsin = dm[:, 4:5]
            tcos = dm[:, 5:6]
            tobj = dm[:, 6:7]
            tcls = dm[:, 7:] if nc > 0 else None

            # grid
            gx, gy = make_grid(H, W, dm.device)

            # map to image coords
            sx = tx.sigmoid()
            sy = ty.sigmoid()
            pw = tw.float().exp() * float(s)
            ph = th.float().exp() * float(s)
            ang = torch.atan2(tsin, tcos)

            cx = (sx + gx) * float(s)
            cy = (sy + gy) * float(s)

            # le-90 canonicalisation
            mask = (pw < ph)
            pw, ph = torch.where(mask, ph, pw), torch.where(mask, pw, ph)
            pi_over_2 = ang.new_tensor(math.pi / 2.0)
            pi = ang.new_tensor(math.pi)
            ang = ang + mask.to(ang.dtype) * pi_over_2
            ang = torch.remainder(ang + pi_over_2, pi) - pi_over_2

            # scores / labels
            if (nc <= 0) or (tcls is None) or (tcls.shape[1] == 0):
                scores_map = tobj.sigmoid()  # [B,1,H,W]
                labels_map = torch.zeros_like(scores_map, dtype=torch.long, device=dm.device)
            else:
                cls = tcls.sigmoid()  # [B,nc,H,W]
                if multi_label:
                    scores_map = cls * tobj.sigmoid()  # [B,nc,H,W]
                    labels_map = None
                else:
                    scores_map, labels_map = (cls * tobj.sigmoid()).max(dim=1, keepdim=True)  # [B,1,H,W], [B,1,H,W]

            # flatten per image
            cx = cx.view(B, -1).to(dtype)
            cy = cy.view(B, -1).to(dtype)
            pw = pw.view(B, -1).to(dtype)
            ph = ph.view(B, -1).to(dtype)
            ang = ang.view(B, -1).to(dtype)

            if multi_label and (nc > 0) and (labels_map is None):
                # Multi-label path: per-class thresholds (kept for completeness)
                B_, Cc, H_, W_ = scores_map.shape
                scores_map = scores_map.view(B, Cc, -1)
                for b in range(B):
                    s_b = scores_map[b]  # [nc, HW]
                    keep = s_b > conf_thres
                    if keep.any():
                        cls_idx, pos_idx = keep.nonzero(as_tuple=True)
                        # per-level cap
                        if cls_idx.numel() > max_det_per_level:
                            vals = s_b[cls_idx, pos_idx]
                            topv, topi = torch.topk(vals, k=max_det_per_level, largest=True, sorted=False)
                            cls_idx = cls_idx[topi];
                            pos_idx = pos_idx[topi]
                        out_boxes[b].append(torch.stack([cx[b, pos_idx], cy[b, pos_idx],
                                                         pw[b, pos_idx], ph[b, pos_idx], ang[b, pos_idx]], dim=1))
                        out_scores[b].append(s_b[cls_idx, pos_idx])
                        out_labels[b].append(cls_idx.to(torch.long))
            else:
                scores = scores_map.view(B, -1)
                labels = (labels_map.view(B, -1).long()
                          if (labels_map is not None) else torch.zeros_like(scores, dtype=torch.long, device=dm.device))
                keep = scores > conf_thres
                for b in range(B):
                    k = keep[b].nonzero(as_tuple=False).squeeze(1)
                    if k.numel() == 0:
                        continue
                    # per-level cap so P3 cannot dominate
                    nb = min(int(max_det_per_level), int(k.numel()))
                    topk = torch.topk(scores[b, k], k=nb, largest=True, sorted=False).indices
                    idx = k[topk]
                    out_boxes[b].append(torch.stack([cx[b, idx], cy[b, idx],
                                                     pw[b, idx], ph[b, idx], ang[b, idx]], dim=1))
                    out_scores[b].append(scores[b, idx])
                    out_labels[b].append(labels[b, idx])

        self._decode_once = True

        # pack per-image + light NMS + final global cap
        results = []
        for b in range(B):
            if out_boxes[b]:
                boxes = torch.cat(out_boxes[b], dim=0)
                scores = torch.cat(out_scores[b], dim=0)
                labels = torch.cat(out_labels[b], dim=0)
                # AABB NMS (fast, keeps level diversity while scores are flat)
                if use_nms and boxes.shape[0] > 0:
                    aabb = aabb_from_obb(boxes)
                    keep = nms_aabb(aabb, scores, iou_thres)
                    if keep.numel() > 0:
                        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                # final cap
                if boxes.shape[0] > max_det:
                    topk = torch.topk(scores, k=max_det, largest=True, sorted=False).indices
                    boxes = boxes[topk]
                    scores = scores[topk]
                    labels = labels[topk]
            else:
                boxes = torch.zeros((0, 5), device=device, dtype=dtype)
                scores = torch.zeros((0,), device=device, dtype=dtype)
                labels = torch.zeros((0,), device=device, dtype=torch.long)
            results.append({"boxes": boxes, "scores": scores, "labels": labels})

        return results
