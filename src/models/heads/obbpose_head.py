
# src/models/heads/obbpose_head.py
from __future__ import annotations
from typing import List, Tuple

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
            iou_thres: float = 0.45,  # kept for signature compat
            max_det: int = 300,
            use_nms: bool = True,  # kept for signature compat
            multi_label: bool = False,
            agnostic: bool = False,
            strides=(8, 16, 32),  # ignored if auto stride used
    ):
        """
        Expects per-level channels: [tx,ty,tw,th,sin,cos,obj,(cls...)].
        Auto-inferrs stride per level from imgs.shape[-2:] vs feature HxW.
        Applies le-90 canonicalisation (w>=h, wrap θ to [-π/2, π/2)).
        """
        import torch
        B, _, Hi, Wi = imgs.shape
        assert isinstance(det_maps, (list, tuple)) and len(det_maps) == 3, "expect 3 FPN levels"

        def make_grid(h, w, device):
            yv, xv = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            return xv.view(1, 1, h, w), yv.view(1, 1, h, w)

        # one-time assert gate
        if not hasattr(self, "_decode_once"):
            self._decode_once = False

        out_boxes = [[] for _ in range(B)]
        out_scores = [[] for _ in range(B)]
        out_labels = [[] for _ in range(B)]

        for level_idx, dm in enumerate(det_maps):
            assert dm.dim() == 4, "det map must be (B,C,H,W)"
            Bb, C, H, W = dm.shape
            assert Bb == B
            nc = C - 7
            assert C == 7 + nc, f"Decoder got C={C}, expected 7+nc (got nc={nc})"

            # ---- AUTO-STRIDE (robust to wrong order) ----
            s_h = int(round(Hi / H)) if H > 0 else 1
            s_w = int(round(Wi / W)) if W > 0 else 1
            s = s_w if s_w == s_h else int(round((Hi / max(1, H) + Wi / max(1, W)) * 0.5))
            if not self._decode_once:
                print(f"[DECODE ASSERT] level#{level_idx} fmap={H}x{W}  auto_stride={s}")
            # ---------------------------------------------

            # slices
            tx = dm[:, 0:1]
            ty = dm[:, 1:2]
            tw = dm[:, 2:3]
            th = dm[:, 3:4]
            tsin = dm[:, 4:5]
            tcos = dm[:, 5:6]
            tobj = dm[:, 6:7]
            tcls = dm[:, 7:] if nc > 0 else None

            # base mapping
            sx = tx.sigmoid()
            sy = ty.sigmoid()
            pw = tw.float().exp() * s
            ph = th.float().exp() * s
            ang = torch.atan2(tsin, tcos)

            gx, gy = make_grid(H, W, device=dm.device)
            cx = (sx + gx) * s
            cy = (sy + gy) * s

            # ---- le-90 canonicalisation (AMP-safe) ----
            mask = (pw < ph)
            pw_new = torch.where(mask, ph, pw)
            ph_new = torch.where(mask, pw, ph)
            pw, ph = pw_new, ph_new
            pi_over_2 = ang.new_tensor(3.141592653589793 / 2.0)
            pi = ang.new_tensor(3.141592653589793)
            ang = ang + mask.to(ang.dtype) * pi_over_2
            ang = torch.remainder(ang + pi_over_2, pi) - pi_over_2
            # -------------------------------------------

            # scores / labels
            if (nc <= 0) or (tcls is None) or (tcls.shape[1] == 0):
                scores_map = tobj.sigmoid()
                labels_map = torch.zeros_like(scores_map, dtype=torch.long, device=dm.device)
            else:
                cls = tcls.sigmoid()
                if multi_label:
                    scores_map = cls * tobj.sigmoid()
                    labels_map = None
                else:
                    scores_map, labels_map = (cls * tobj.sigmoid()).max(dim=1, keepdim=True)

            # flatten
            cx = cx.view(B, -1)
            cy = cy.view(B, -1)
            pw = pw.view(B, -1)
            ph = ph.view(B, -1)
            ang = ang.view(B, -1)

            if multi_label and (nc > 0) and (labels_map is None):
                B_, Cc, H_, W_ = scores_map.shape
                scores_map = scores_map.view(B, Cc, -1)
                for b in range(B):
                    s_b = scores_map[b]
                    keep = s_b > conf_thres
                    if keep.any():
                        cls_idx, pos_idx = keep.nonzero(as_tuple=True)
                        if cls_idx.numel() > max_det:
                            topv, topi = torch.topk(s_b[cls_idx, pos_idx], k=max_det, largest=True, sorted=False)
                            cls_idx = cls_idx[topi]
                            pos_idx = pos_idx[topi]
                        out_boxes[b].append(torch.stack([
                            cx[b, pos_idx], cy[b, pos_idx],
                            pw[b, pos_idx], ph[b, pos_idx],
                            ang[b, pos_idx],
                        ], dim=1))
                        out_scores[b].append(s_b[cls_idx, pos_idx])
                        out_labels[b].append(cls_idx.to(torch.long))
            else:
                scores = scores_map.view(B, -1)
                labels = (labels_map.view(B, -1).long()
                          if (labels_map is not None) else torch.zeros_like(scores, dtype=torch.long, device=dm.device))
                keep = scores > conf_thres
                for b in range(B):
                    k = keep[b].nonzero(as_tuple=False).squeeze(1)
                    if k.numel() == 0: continue
                    nb = min(int(max_det), int(k.numel()))
                    topk = torch.topk(scores[b, k], k=nb, largest=True, sorted=False).indices
                    idx = k[topk]
                    out_boxes[b].append(torch.stack([
                        cx[b, idx], cy[b, idx],
                        pw[b, idx], ph[b, idx],
                        ang[b, idx],
                    ], dim=1))
                    out_scores[b].append(scores[b, idx])
                    out_labels[b].append(labels[b, idx])

        self._decode_once = True

        # pack per-image
        results = []
        for b in range(B):
            if out_boxes[b]:
                boxes = torch.cat(out_boxes[b], dim=0)
                scores = torch.cat(out_scores[b], dim=0)
                labels = torch.cat(out_labels[b], dim=0)
                if boxes.shape[0] > max_det:
                    topk = torch.topk(scores, k=max_det, largest=True, sorted=False).indices
                    boxes = boxes[topk]
                    scores = scores[topk]
                    labels = labels[topk]
            else:
                boxes = torch.zeros((0, 5), device=imgs.device, dtype=imgs.dtype)
                scores = torch.zeros((0,), device=imgs.device, dtype=imgs.dtype)
                labels = torch.zeros((0,), device=imgs.device, dtype=torch.long)
            results.append({"boxes": boxes, "scores": scores, "labels": labels})
        return results
