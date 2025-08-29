
from __future__ import annotations

import math
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
from torchvision.ops import nms as aabb_nms


class OBBPoseHead(nn.Module):
    """
    YOLO-style head for Oriented Bounding Boxes + minimal keypoint heatmaps.

    Per-level det map channel order:
      [ tx, ty,
        dfl_w[0..reg_max], dfl_h[0..reg_max],
        ang_logit, obj, (cls0..cls{nc-1}) ]

    - Angles are represented with a single logit and decoded to radians in (-pi/2, pi/2].
    - Width/height use Distribution Focal Loss (DFL) bins; decoded via expectation.
    - Objectness/cls are raw logits.
    - Decoding combines (obj * best-class) for confidence.
    """
    """
    def __init__(self, ch: Tuple[int, int, int], num_classes: int, reg_max: int = 24) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self.nbins = self.reg_max + 1

        c3, c4, c5 = [int(x) for x in ch]
        # tx, ty, DFLw, DFLh, ang, obj, classes
        det_out = 2 + 2 * self.nbins + 1 + 1 + self.num_classes
        kpt_out = 3  # placeholder head to match interfaces

        block3 = [nn.Conv2d(c3, c3, 3, 1, 1), nn.BatchNorm2d(c3), nn.SiLU(inplace=False)]
        block4 = [nn.Conv2d(c4, c4, 3, 1, 1), nn.BatchNorm2d(c4), nn.SiLU(inplace=False)]
        block5 = [nn.Conv2d(c5, c5, 3, 1, 1), nn.BatchNorm2d(c5), nn.SiLU(inplace=False)]

        self.det3 = nn.Sequential(*block3, nn.Conv2d(c3, det_out, 1, 1, 0))
        self.det4 = nn.Sequential(*block4, nn.Conv2d(c4, det_out, 1, 1, 0))
        self.det5 = nn.Sequential(*block5, nn.Conv2d(c5, det_out, 1, 1, 0))

        self.kp3 = nn.Sequential(*block3, nn.Conv2d(c3, kpt_out, 1, 1, 0))
        self.kp4 = nn.Sequential(*block4, nn.Conv2d(c4, kpt_out, 1, 1, 0))
        self.kp5 = nn.Sequential(*block5, nn.Conv2d(c5, kpt_out, 1, 1, 0))

        self.register_buffer("dfl_bins", torch.arange(self.nbins, dtype=torch.float32).view(1, self.nbins, 1, 1),
                             persistent=False)

        self._assert_once = False
        self._ang_print = False
        self._decode_print = False

        # bias init like YOLO: low obj, slightly low cls
        self._init_head_bias(self.det3)
        self._init_head_bias(self.det4)
        self._init_head_bias(self.det5)

    def _init_head_bias(self, head: nn.Sequential) -> None:
        last = head[-1]
        assert isinstance(last, nn.Conv2d)
        with torch.no_grad():
            if last.bias is None:
                last.bias = nn.Parameter(torch.zeros(last.out_channels, device=last.weight.device))
            last.bias.zero_()
            obj_idx = 2 + 2 * self.nbins + 1
            if last.bias.numel() > obj_idx:
                last.bias[obj_idx] = -3.0  # ~obj prior 0.047
            cls_start = obj_idx + 1
            if last.bias.numel() > cls_start:
                last.bias[cls_start:] = -2.0

    def forward(self, feats: List[torch.Tensor]):
        assert isinstance(feats, (list, tuple)) and len(feats) == 3, "OBBPoseHead.forward expects [P3,P4,P5]"
        p3, p4, p5 = feats
        det_maps = [self.det3(p3), self.det4(p4), self.det5(p5)]
        kpt_maps = [self.kp3(p3), self.kp4(p4), self.kp5(p5)]

        if not self._assert_once:
            expected = 2 + 2 * self.nbins + 1 + 1 + self.num_classes
            for i, lvl in enumerate(("P3", "P4", "P5")):
                c = int(det_maps[i].shape[1])
                assert c == expected, f"{lvl}: channels={c}, expected={expected} (tx,ty,DFLw,DFLh,ang,obj,cls)"
            self._assert_once = True
        if not self._ang_print:
            print("[ASSERT] angle uses single‑logit (YOLO‑style).")
            self._ang_print = True
        return det_maps, kpt_maps

    @staticmethod
    def _le90(w: torch.Tensor, h: torch.Tensor, ang: torch.Tensor):
        """Canonicalize to w>=h and wrap angle to (-pi/2, pi/2]."""
        swap = w < h
        w2 = torch.where(swap, h, w)
        h2 = torch.where(swap, w, h)
        ang2 = torch.where(swap, ang + math.pi / 2.0, ang)
        ang2 = ((ang2 + math.pi / 2.0) % math.pi) - math.pi / 2.0
        return w2, h2, ang2

    @torch.no_grad()
    def decode_obb_from_pyramids(
        self,
        det_maps: List[torch.Tensor],
        imgs: torch.Tensor,
        *,
        strides: Tuple[int, int, int] = (8, 16, 32),
        score_thr: float = 0.25,
        iou_thres: float = 0.5,
        max_det: int = 100,
        use_nms: bool = True,
        multi_label: bool = False,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Decode maps to oriented boxes. Angles remain in radians until NMS/IoU with MMCV
        (which expects **degrees**). We convert only for NMS and keep original radians.
        """
        B, _, Himg, Wimg = imgs.shape
        device = imgs.device
        nb = self.nbins

        if not self._decode_print:
            self._decode_print = True
            print(f"[DECODE ASSERT] DFL decode | reg_max={self.reg_max} nbins={self.nbins} "
                  f"score_thr={score_thr:.3f} use_nms={use_nms} iou={iou_thres:.2f}")

        out: List[Dict[str, torch.Tensor]] = []

        for b in range(B):
            boxes_all, scores_all, labels_all = [], [], []
            for (dm, s) in zip(det_maps, strides):
                _, C, H, W = dm.shape
                idx = 0
                tx = dm[b, idx:idx + 1]; idx += 1
                ty = dm[b, idx:idx + 1]; idx += 1
                dflw = dm[b, idx:idx + nb]; idx += nb
                dflh = dm[b, idx:idx + nb]; idx += nb
                ang_logit = dm[b, idx:idx + 1]; idx += 1
                obj = dm[b, idx:idx + 1]; idx += 1
                cls = dm[b, idx:] if C > idx else None

                N = H * W
                obj_p = obj.sigmoid().view(-1)  # [N]

                # combine confidence = obj * best-class (or obj if single-class head)
                if (cls is not None) and (cls.numel() > 0):
                    cp = cls.sigmoid()
                    if cp.dim() == 3:  # [C,H,W] -> [N,C]
                        cp = cp.permute(1, 2, 0).reshape(-1, cp.shape[0])
                    else:
                        cp = cp.reshape(-1, cp.shape[-1])
                    cls_best = cp.max(dim=1).values
                    conf = obj_p * cls_best
                else:
                    conf = obj_p

                keep = (conf > score_thr).nonzero(as_tuple=False).squeeze(1)
                if keep.numel() == 0:
                    continue

                PRE_NMS_TOPK = 1000
                if keep.numel() > PRE_NMS_TOPK:
                    _, topi = conf.index_select(0, keep).topk(PRE_NMS_TOPK, largest=True, sorted=False)
                    keep = keep[topi]

                # indices
                iy = (keep // W).to(tx.dtype)
                ix = (keep %  W).to(tx.dtype)

                # centers
                sx = tx[0].view(-1).index_select(0, keep).sigmoid()
                sy = ty[0].view(-1).index_select(0, keep).sigmoid()
                cx = (ix + sx) * float(s)
                cy = (iy + sy) * float(s)

                # sizes from DFL
                dflw_k = dflw.reshape(nb, N).index_select(1, keep).float()
                dflh_k = dflh.reshape(nb, N).index_select(1, keep).float()
                bins = torch.arange(nb, device=device, dtype=dflw_k.dtype).view(nb, 1)
                pw = (torch.softmax(dflw_k, dim=0) * bins).sum(0) * float(s)
                ph = (torch.softmax(dflh_k, dim=0) * bins).sum(0) * float(s)

                # angle (radians in (-pi/2, pi/2])
                ang = ang_logit.view(-1).index_select(0, keep).sigmoid() * math.pi - (math.pi / 2.0)

                # canonical LE-90
                pw, ph, ang = self._le90(pw.to(obj.dtype), ph.to(obj.dtype), ang)

                # labels/scores
                obj_k = obj_p.index_select(0, keep)
                if (cls is None) or (cls.shape[0] == 0):
                    scores = conf.index_select(0, keep)
                    labels = torch.zeros_like(scores, dtype=torch.long, device=device)
                elif multi_label:
                    cls_b = cls.sigmoid().view(self.num_classes, N).index_select(1, keep)
                    s_mat = cls_b * obj_k.unsqueeze(0)
                    c_idx, p_idx = (s_mat > score_thr).nonzero(as_tuple=True)
                    if p_idx.numel() == 0:
                        cls_scores, cls_labels = cls_b.max(dim=0)
                        scores = cls_scores * obj_k
                        labels = cls_labels.to(torch.long)
                    else:
                        scores = s_mat[c_idx, p_idx]
                        labels = c_idx.to(torch.long)
                        cx = cx.index_select(0, p_idx)
                        cy = cy.index_select(0, p_idx)
                        pw = pw.index_select(0, p_idx)
                        ph = ph.index_select(0, p_idx)
                        ang = ang.index_select(0, p_idx)
                else:
                    cls_b = cls.sigmoid().view(self.num_classes, N).index_select(1, keep)
                    cls_scores, cls_labels = cls_b.max(dim=0)
                    scores = cls_scores * obj_k
                    labels = cls_labels.to(torch.long)

                boxes = torch.stack([cx, cy, pw.clamp_(1.0), ph.clamp_(1.0), ang], dim=1)
                boxes_all.append(boxes)
                scores_all.append(scores)
                labels_all.append(labels)

            if boxes_all:
                boxes = torch.cat(boxes_all, dim=0)
                scores = torch.cat(scores_all, dim=0)
                labels = torch.cat(labels_all, dim=0)

                # NMS (MMCV expects DEGREES)
                if use_nms and boxes.numel() > 0:
                    try:
                        from mmcv.ops import nms_rotated as mmcv_nms_rotated
                        b_deg = boxes.clone()
                        b_deg[:, 4] = torch.rad2deg(b_deg[:, 4])
                        # mmcv returns (dets, keep_idx)
                        _, keep_idx = mmcv_nms_rotated(b_deg, scores, iou_thres, labels=None, clockwise=False)
                        if keep_idx.numel() > 0:
                            boxes = boxes.index_select(0, keep_idx)
                            scores = scores.index_select(0, keep_idx)
                            labels = labels.index_select(0, keep_idx)
                    except Exception:
                        # fallback AABB NMS if mmcv isn't available at runtime
                        x1 = boxes[:, 0] - boxes[:, 2] * 0.5
                        y1 = boxes[:, 1] - boxes[:, 3] * 0.5
                        x2 = boxes[:, 0] + boxes[:, 2] * 0.5
                        y2 = boxes[:, 1] + boxes[:, 3] * 0.5
                        abox = torch.stack([x1, y1, x2, y2], dim=1)
                        keep_idx = aabb_nms(abox, scores, iou_thres)
                        boxes = boxes.index_select(0, keep_idx)
                        scores = scores.index_select(0, keep_idx)
                        labels = labels.index_select(0, keep_idx)

                # cap
                if boxes.shape[0] > max_det:
                    topk = scores.topk(max_det, largest=True, sorted=False).indices
                    boxes = boxes.index_select(0, topk)
                    scores = scores.index_select(0, topk)
                    labels = labels.index_select(0, topk)
            else:
                boxes = torch.zeros((0, 5), device=device)
                scores = torch.zeros((0,), device=device)
                labels = torch.zeros((0,), dtype=torch.long, device=device)

            out.append({"boxes": boxes, "scores": scores, "labels": labels})

        return out
