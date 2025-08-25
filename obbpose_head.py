
# src/models/heads/obbpose_head.py
from __future__ import annotations
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import math

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
        det_maps: List[torch.Tensor],
        imgs: torch.Tensor,
        *,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
        use_nms: bool = True,
        multi_label: bool = False,
        agnostic: bool = False,
        strides=(8, 16, 32),
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Map (tx,ty,tw,th,sin,cos,obj,[cls...]) back to (cx,cy,w,h,θ) per image.

        Decoding per level with stride s:
            cx = (sigmoid(tx) + grid_x) * s
            cy = (sigmoid(ty) + grid_y) * s
            w  = exp(tw) * s
            h  = exp(th) * s
            θ  = atan2(sin, cos)   # radians
            score = σ(obj) * (σ(cls[c]) if multi-class else 1)

        Returns list (len=B) of dicts: boxes(N,5), scores(N), labels(N)
        """
        device = imgs.device
        B = imgs.shape[0]
        assert isinstance(det_maps, (list, tuple)) and len(det_maps) == 3, "expect 3 levels"

        def make_grid(h, w, device):
            yv, xv = torch.meshgrid(torch.arange(h, device=device),
                                    torch.arange(w, device=device),
                                    indexing="ij")
            return xv.view(1, 1, h, w), yv.view(1, 1, h, w)

        out_boxes = [[] for _ in range(B)]
        out_scores = [[] for _ in range(B)]
        out_labels = [[] for _ in range(B)]

        for dm, s in zip(det_maps, strides):
            assert dm.dim() == 4, "det map must be (B,C,H,W)"
            Bc, C, H, W = dm.shape
            assert Bc == B
            expected = 7 + self.num_classes
            assert C == expected, f"Decoder got C={C}, expected {expected} (=7+nc). Check head channels."

            tx = dm[:, 0:1]; ty = dm[:, 1:2]
            tw = dm[:, 2:3]; th = dm[:, 3:4]
            tsin = dm[:, 4:5]; tcos = dm[:, 5:6]
            tobj = dm[:, 6:7]
            tcls = dm[:, 7:] if self.num_classes > 0 else None

            sx = tx.sigmoid(); sy = ty.sigmoid()
            pw = tw.float().exp() * s
            ph = th.float().exp() * s
            ang = torch.atan2(tsin, tcos)
            obj = tobj.sigmoid()

            gx, gy = make_grid(H, W, device=device)
            cx = (sx + gx) * s
            cy = (sy + gy) * s

            
# --- Canonicalize to le-90: ensure w >= h and theta in [-pi/2, pi/2) ---
mask = pw < ph
if mask.any():
    pw, ph = torch.where(mask, ph, pw), torch.where(mask, pw, ph)
    ang = ang + (mask.to(ang.dtype) * (math.pi / 2.0))
# Wrap angle to [-pi/2, pi/2)
ang = torch.remainder(ang + (math.pi / 2.0), math.pi) - (math.pi / 2.0)
if self.num_classes <= 1 or tcls is None or tcls.shape[1] == 0:
                scores_map = obj
                labels_map = torch.zeros_like(scores_map, dtype=torch.long, device=device)
            else:
                cls = tcls.sigmoid()
                if multi_label:
                    scores_map = cls * obj
                    labels_map = None
                else:
                    scores_map, labels_map = (cls * obj).max(dim=1, keepdim=True)

            # flatten
            cx = cx.view(B, -1); cy = cy.view(B, -1)
            pw = pw.view(B, -1); ph = ph.view(B, -1)
            ang = ang.view(B, -1)

            if multi_label and (self.num_classes > 1) and (labels_map is None):
                B_, Cc, H_, W_ = scores_map.shape
                scores_map = scores_map.view(B, Cc, -1)
                for b in range(B):
                    s_b = scores_map[b]
                    keep = s_b > conf_thres
                    if keep.any():
                        cls_idx, pos_idx = keep.nonzero(as_tuple=True)
                        if cls_idx.numel() > max_det:
                            topv, topi = torch.topk(s_b[cls_idx, pos_idx], k=max_det, largest=True, sorted=False)
                            cls_idx = cls_idx[topi]; pos_idx = pos_idx[topi]
                        out_boxes[b].append(torch.stack([cx[b, pos_idx], cy[b, pos_idx],
                                                         pw[b, pos_idx], ph[b, pos_idx],
                                                         ang[b, pos_idx]], dim=1))
                        out_scores[b].append(s_b[cls_idx, pos_idx])
                        out_labels[b].append(cls_idx.to(torch.long))
            else:
                scores = scores_map.view(B, -1)
                if self.num_classes <= 1:
                    labels = torch.zeros_like(scores, dtype=torch.long, device=device)
                else:
                    labels = labels_map.view(B, -1).long()

                keep = scores > conf_thres
                for b in range(B):
                    k = keep[b].nonzero(as_tuple=False).squeeze(1)
                    if k.numel() == 0:
                        continue
                    nb = min(int(max_det), int(k.numel()))
                    topk = torch.topk(scores[b, k], k=nb, largest=True, sorted=False).indices
                    idx = k[topk]
                    out_boxes[b].append(torch.stack([cx[b, idx], cy[b, idx],
                                                     pw[b, idx], ph[b, idx],
                                                     ang[b, idx]], dim=1))
                    out_scores[b].append(scores[b, idx])
                    out_labels[b].append(labels[b, idx])

        results = []
        for b in range(B):
            if out_boxes[b]:
                boxes = torch.cat(out_boxes[b], dim=0)
                scores = torch.cat(out_scores[b], dim=0)
                labels = torch.cat(out_labels[b], dim=0)
                if boxes.shape[0] > max_det:
                    topk = torch.topk(scores, k=max_det, largest=True, sorted=False).indices
                    boxes = boxes[topk]; scores = scores[topk]; labels = labels[topk]
            else:
                boxes = torch.zeros((0, 5), device=device, dtype=imgs.dtype)
                scores = torch.zeros((0,), device=device, dtype=imgs.dtype)
                labels = torch.zeros((0,), device=device, dtype=torch.long)
            results.append({"boxes": boxes, "scores": scores, "labels": labels})
        return results