# src/models/losses/td_obb_kpt1_loss.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TDOBBWKpt1Criterion(nn.Module):
    """
    Detection + single-keypoint criterion for YOLO11-style OBB + top-down keypoint.

    Inputs
    ------
    det_maps : list[Tensor]
        Three tensors [(B, C, H, W), ...] for strides (8, 16, 32).
        Channels C = 7 + nc laid out as: [tx, ty, tw, th, sin, cos, obj, (cls...)].
    feats : list[Tensor]
        PAN-FPN outputs [n3, d4, d5] used for ROI pooling (same strides 8/16/32).
    batch : dict
        Either:
          • {'bboxes': list(Ti,5 radians), 'labels': list(Ti,), 'kpts': list(Ti,2 px)}
          • or YOLO-style 'targets' tensor (M, 8/9): [bix, cls, cx,cy,w,h,ang(rad), kpx,kpy]
    model : nn.Module
        Must expose .kpt_from_obbs(feats, obb_list, ...) and .roi (with out_size, feat_down).

    Returns
    -------
    total_loss : Tensor (scalar)
    logs : dict[str, float]
    """

    def __init__(
        self,
        num_classes: int,
        strides: Sequence[int] = (8, 16, 32),
        lambda_box: float = 1.0,
        lambda_obj: float = 1.0,
        lambda_ang: float = 0.2,
        lambda_cls: float = 0.5,
        lambda_kpt: float = 0.5,
        kpt_freeze_epochs: int = 0,
        kpt_warmup_epochs: int = 0,
        level_boundaries: Tuple[float, float] = (64.0, 128.0),  # px thresholds to route GTs to p3/p4/p5
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.strides = tuple(int(s) for s in strides)
        self.lambda_box = float(lambda_box)
        self.lambda_obj = float(lambda_obj)
        self.lambda_ang = float(lambda_ang)
        self.lambda_cls = float(lambda_cls)
        self.lambda_kpt = float(lambda_kpt)
        self.kpt_freeze_epochs = int(kpt_freeze_epochs)
        self.kpt_warmup_epochs = int(kpt_warmup_epochs)
        if len(level_boundaries) != 2:
            raise ValueError("level_boundaries must be a (low, high) tuple in pixels.")
        self.level_boundaries = (float(level_boundaries[0]), float(level_boundaries[1]))

        self.bce_logits = nn.BCEWithLogitsLoss(reduction="mean")
        self.smoothl1 = nn.SmoothL1Loss(reduction="mean")

    # ---------------------- Public ----------------------

    def forward(
        self,
        det_maps: List[torch.Tensor] | torch.Tensor,
        feats: Optional[List[torch.Tensor]],
        batch: Dict[str, Any],
        model: Optional[nn.Module] = None,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Normalize inputs
        if isinstance(det_maps, torch.Tensor):
            det_maps = [det_maps]
        assert len(det_maps) == 3, f"Expected 3 pyramid levels, got {len(det_maps)}"
        device = det_maps[0].device
        B = det_maps[0].shape[0]

        # Split batch into per-image lists
        boxes_list, labels_list, kpts_list = self._split_targets(batch, B, device)

        # Assign GTs to levels and build positive indices
        pos_targets = self._build_pos_targets(det_maps, boxes_list, labels_list)

        # Detection losses
        l_box, l_obj, l_ang, l_cls, total_pos = self._det_losses(det_maps, pos_targets)

        if total_pos == 0:
            total_det = self.lambda_obj * l_obj
        else:
            total_det = (
                self.lambda_box * (l_box / total_pos) +
                self.lambda_obj * l_obj +
                self.lambda_ang * (l_ang / total_pos) +
                (self.lambda_cls * (l_cls / total_pos) if self.num_classes > 1 else 0.0)
            )

        # Keypoint loss (top-down) if enabled
        l_kpt = torch.zeros((), device=device)
        kpt_pos = 0
        kpt_scale = 0.0
        if (model is not None) and (feats is not None) and (epoch >= self.kpt_freeze_epochs):
            l_kpt, kpt_pos = self._keypoint_loss(model, feats, boxes_list, kpts_list)
            if self.kpt_warmup_epochs > 0 and epoch < (self.kpt_freeze_epochs + self.kpt_warmup_epochs):
                ramp = float(epoch - self.kpt_freeze_epochs + 1) / float(self.kpt_warmup_epochs)
                kpt_scale = max(0.0, min(1.0, ramp))
            else:
                kpt_scale = 1.0

        total = total_det + (self.lambda_kpt * kpt_scale * l_kpt)

        logs = {
            "loss_box": float(((l_box / max(1, total_pos))).detach().item() if total_pos else 0.0),
            "loss_obj": float(l_obj.detach().item()),
            "loss_ang": float(((l_ang / max(1, total_pos))).detach().item() if total_pos else 0.0),
            "loss_cls": float(((l_cls / max(1, total_pos))).detach().item() if (self.num_classes > 1 and total_pos) else 0.0),
            "loss_kpt": float((self.lambda_kpt * kpt_scale * l_kpt).detach().item()),
            "num_pos": float(total_pos),
            "kpt_pos": float(kpt_pos),
        }
        return total, logs

    # ---------------------- Helpers (internal) ----------------------

    def _split_targets(
        self,
        batch: Dict[str, Any],
        B: int,
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Accepts either per-image lists (bboxes/labels/kpts) or a single YOLO-style 'targets' tensor."""
        if "bboxes" in batch and "labels" in batch:
            boxes_list = [b.to(device=device, dtype=torch.float32) for b in batch["bboxes"]]
            labels_list = [l.to(device=device, dtype=torch.long) for l in batch["labels"]]
            kpts_raw = batch.get("kpts", [None] * B)
            kpts_list = [ (k.to(device=device, dtype=torch.float32) if k is not None else torch.empty((0,2), device=device))
                          for k in kpts_raw ]
            return boxes_list, labels_list, kpts_list

        t = batch.get("targets", None)
        if t is None or not torch.is_tensor(t):
            empty_b = [torch.zeros((0,5), device=device) for _ in range(B)]
            empty_l = [torch.zeros((0,), dtype=torch.long, device=device) for _ in range(B)]
            empty_k = [torch.empty((0,2), device=device) for _ in range(B)]
            return empty_b, empty_l, empty_k

        t = t.to(device=device, dtype=torch.float32)  # (M, 8/9)
        has_kp = t.shape[1] >= 9
        boxes_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        kpts_list: List[torch.Tensor] = []
        for b in range(B):
            sel = (t[:, 0].long() == b)
            tb = t[sel]
            if tb.numel() == 0:
                boxes_list.append(torch.zeros((0,5), device=device))
                labels_list.append(torch.zeros((0,), dtype=torch.long, device=device))
                kpts_list.append(torch.empty((0,2), device=device))
                continue
            labels_list.append(tb[:, 1].long())
            boxes_list.append(tb[:, 2:7])  # cx,cy,w,h,ang(rad)
            if has_kp:
                kpts_list.append(tb[:, 7:9])
            else:
                kpts_list.append(torch.empty((0,2), device=device))
        return boxes_list, labels_list, kpts_list

    def _assign_levels(self, w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Route objects by max side length to p3/p4/p5 using configured thresholds (px)."""
        low, high = self.level_boundaries
        size = torch.max(w, h)
        return torch.where(size < low, 0, torch.where(size < high, 1, 2))  # 0->p3, 1->p4, 2->p5

    def _build_pos_targets(
        self,
        det_maps: List[torch.Tensor],
        boxes_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
    ) -> List[List[Tuple[int, int, int, Dict[str, float]]]]:
        """For each level, produce positives: (bix, j, i, target dict)."""
        pos_targets: List[List[Tuple[int, int, int, Dict[str, float]]]] = [[] for _ in det_maps]

        for b, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
            if boxes.numel() == 0:
                continue
            cx, cy, w, h, ang = boxes.t()
            levels = self._assign_levels(w, h)

            for n in range(boxes.shape[0]):
                li = int(levels[n].item())
                stride = self.strides[li]
                Hf, Wf = det_maps[li].shape[-2], det_maps[li].shape[-1]
                gx, gy = cx[n] / stride, cy[n] / stride
                i = int(torch.clamp(gx.floor(), 0, Wf - 1).item())
                j = int(torch.clamp(gy.floor(), 0, Hf - 1).item())
                tx = float(gx - i)
                ty = float(gy - j)
                tw = float(torch.log(torch.clamp(w[n] / stride, min=1e-6)))
                th = float(torch.log(torch.clamp(h[n] / stride, min=1e-6)))
                s = float(torch.sin(ang[n]))
                c = float(torch.cos(ang[n]))
                cls = int(labels[n].item()) if labels.numel() else 0
                pos_targets[li].append((b, j, i, {"tx": tx, "ty": ty, "tw": tw, "th": th, "sin": s, "cos": c, "cls": cls}))

        return pos_targets

    def _det_losses(
        self,
        det_maps: List[torch.Tensor],
        pos_targets: List[List[Tuple[int, int, int, Dict[str, float]]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Compute detection losses for all levels and sum them up."""
        device = det_maps[0].device
        l_box = torch.zeros((), device=device)
        l_obj = torch.zeros((), device=device)
        l_ang = torch.zeros((), device=device)
        l_cls = torch.zeros((), device=device) if self.num_classes > 1 else torch.zeros((), device=device)

        total_pos = 0
        for li, dm in enumerate(det_maps):
            B, C, Hf, Wf = dm.shape
            tx = dm[:, 0]; ty = dm[:, 1]
            tw = dm[:, 2]; th = dm[:, 3]
            s  = dm[:, 4]; c  = dm[:, 5]
            obj = dm[:, 6]
            cls_logits = dm[:, 7:] if C > 7 else None

            # Objectness targets: 1 for positive cells, else 0
            obj_tgt = torch.zeros_like(obj)

            for (b, j, i, tgt) in pos_targets[li]:
                obj_tgt[b, j, i] = 1.0
                total_pos += 1

                # bbox regression (use activated predictions)
                p_tx = torch.sigmoid(tx[b, j, i])
                p_ty = torch.sigmoid(ty[b, j, i])
                p_tw = torch.exp(tw[b, j, i])
                p_th = torch.exp(th[b, j, i])
                l_box = l_box + self.smoothl1(p_tx, torch.tensor(tgt["tx"], device=device))
                l_box = l_box + self.smoothl1(p_ty, torch.tensor(tgt["ty"], device=device))
                l_box = l_box + self.smoothl1(p_tw, torch.tensor(math.exp(tgt["tw"]), device=device))
                l_box = l_box + self.smoothl1(p_th, torch.tensor(math.exp(tgt["th"]), device=device))

                # angle (sin, cos) after tanh
                p_s = torch.tanh(s[b, j, i])
                p_c = torch.tanh(c[b, j, i])
                l_ang = l_ang + self.smoothl1(p_s, torch.tensor(tgt["sin"], device=device))
                l_ang = l_ang + self.smoothl1(p_c, torch.tensor(tgt["cos"], device=device))

                # classification (multi-label BCE per class)
                if self.num_classes > 1 and cls_logits is not None:
                    y = torch.zeros((self.num_classes,), device=device)
                    y[tgt["cls"]] = 1.0
                    l_cls = l_cls + F.binary_cross_entropy_with_logits(cls_logits[b, :, j, i], y)

            l_obj = l_obj + F.binary_cross_entropy_with_logits(obj, obj_tgt)

        return l_box, l_obj, l_ang, l_cls, total_pos

    def _keypoint_loss(
        self,
        model: nn.Module,
        feats: List[torch.Tensor],
        boxes_list: List[torch.Tensor],
        kpts_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        """Compute top-down keypoint loss using GT OBBs to form ROIs on P3 (/8)."""
        device = feats[0].device

        # Build OBB list (degrees) for ROI
        obb_list: List[torch.Tensor] = []
        total_rois = 0
        for b, ob in enumerate(boxes_list):
            if ob is None or ob.numel() == 0:
                obb_list.append(torch.zeros((0,5), device=device))
                continue
            ob = ob.clone()
            ob[:, 4] = torch.rad2deg(ob[:, 4])
            obb_list.append(ob)
            total_rois += ob.shape[0]

        if total_rois == 0:
            return torch.zeros((), device=device), 0

        # Run top-down head
        uv_pred, metas = model.kpt_from_obbs(
            feats, obb_list, scores_list=None,
            topk=total_rois, chunk=128, score_thresh=0.0
        )
        n_rois = int(uv_pred.shape[0]) if torch.is_tensor(uv_pred) else 0
        if n_rois == 0:
            return torch.zeros((), device=device), 0

        roi = getattr(model, "roi", None)
        S = int(getattr(roi, "out_size", 64))
        feat_down = float(getattr(roi, "feat_down", 8))

        # Targets (crop px)
        uv_tgt = torch.zeros_like(uv_pred, device=device)
        valid = torch.zeros((n_rois,), dtype=torch.bool, device=device)

        idx = 0
        for m in metas:
            b = int(m.get("b", m.get("bix", 0)))
            oi = int(m.get("oi", idx))  # per-image object idx (rot_roi adds this); fallback to sequential
            M = m["M"].to(device=uv_pred.device, dtype=uv_pred.dtype)
            if b >= len(kpts_list) or kpts_list[b] is None or kpts_list[b].numel() == 0:
                idx += 1; continue
            if oi < 0 or oi >= kpts_list[b].shape[0]:
                idx += 1; continue

            uv_px = self._img_kpts_to_crop_uv(kpts_list[b][oi:oi+1, :], M, feat_down)  # (1,2) in px
            uv_px = torch.clamp(uv_px, 0, S - 1)
            uv_tgt[idx] = uv_px[0]
            valid[idx] = True
            idx += 1

        vcnt = int(valid.sum().item())
        if vcnt == 0:
            return torch.zeros((), device=device), 0

        uv_pred_px = uv_pred * float(S - 1)  # scale [0,1] -> crop px
        l_kpt = F.smooth_l1_loss(uv_pred_px[valid], uv_tgt[valid], reduction="mean")
        return l_kpt, vcnt

    # ---------- math helpers ----------

    @staticmethod
    def _img_kpts_to_crop_uv(kpts_img: torch.Tensor, M: torch.Tensor, feat_down: float) -> torch.Tensor:
        """
        Map image keypoints (x,y) in px -> crop (u,v) in px using crop->feature affine M (2x3).
        We first map image px -> feature px by dividing by feat_down, then solve (u,v) from M.
        """
        # M maps crop px -> feature px:
        # [x_f]   [A  B  C][u]
        # [y_f] = [A2 B2 C2][v]
        A, Bv, C = M[0, 0], M[0, 1], M[0, 2]
        A2, B2, C2 = M[1, 0], M[1, 1], M[1, 2]
        det = A * B2 - Bv * A2
        det = det if torch.is_tensor(det) else torch.tensor(det, dtype=M.dtype, device=M.device)
        inv = torch.stack([
            torch.stack([ B2, -Bv, Bv*C2 - B2*C ], dim=0),
            torch.stack([-A2,  A,  A2*C - A*C2], dim=0)
        ], dim=0) / det  # (2,3)

        xy = kpts_img.to(dtype=M.dtype, device=M.device) / float(feat_down)
        ones = torch.ones((xy.shape[0], 1), dtype=M.dtype, device=M.device)
        uv = (inv @ torch.cat([xy, ones], dim=1).t()).t()  # (N,2)
        return uv