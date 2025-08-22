# src/models/losses/td_obb_kpt1_loss.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TDOBBWKpt1Criterion(nn.Module):
    """
    YOLO11-style criterion for Oriented Boxes (single-class or multi-class) + top-down 1-keypoint.

    Key details preserved from the working file:
      - 3-level (P3/P4/P5) target assignment by object size with adjacent-level duplication near thresholds
      - Optional 3×3 neighbor positives around the main cell (neighbor_range=1 -> 3×3)
      - Angle predicted as (sin, cos), normalized before loss
      - Top-down keypoint supervision via model.kpt_from_obbs() using P3 crops
      - Same public API and parameter names so Trainer/train.py do not need to change
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
        level_boundaries: Tuple[float, float] = (64.0, 128.0),  # px thresholds for p3/p4/p5 routing
        neighbor_cells: bool = True,
        neighbor_range: int = 1,             # radius 1 → 3x3 window
        adjacent_level_margin: float = 16.0, # also assign to adjacent level when near boundary
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.strides = tuple(int(s) for s in strides)
        assert len(self.strides) == 3, "Expected three strides for P3/P4/P5"
        self.lambda_box = float(lambda_box)
        self.lambda_obj = float(lambda_obj)
        self.lambda_ang = float(lambda_ang)
        self.lambda_cls = float(lambda_cls)
        self.lambda_kpt = float(lambda_kpt)
        self.kpt_freeze_epochs = int(kpt_freeze_epochs)
        self.kpt_warmup_epochs = int(kpt_warmup_epochs)
        if len(level_boundaries) != 2:
            raise ValueError("level_boundaries must be (low, high) in pixels.")
        self.level_boundaries = (float(level_boundaries[0]), float(level_boundaries[1]))
        self.neighbor_cells = bool(neighbor_cells)
        self.neighbor_range = int(neighbor_range)
        self.adjacent_level_margin = float(adjacent_level_margin)

        self.smoothl1 = nn.SmoothL1Loss(reduction="mean")

    # ---------------------- public ----------------------

    def forward(
        self,
        det_maps: List[torch.Tensor] | torch.Tensor,
        feats: Optional[List[torch.Tensor]],
        batch: Dict[str, Any],
        model: Optional[torch.nn.Module] = None,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # Normalize input to list of 3 feature maps
        if isinstance(det_maps, torch.Tensor):
            det_maps = [det_maps]
        assert len(det_maps) == 3, f"Expected 3 pyramid levels, got {len(det_maps)}"
        device = det_maps[0].device
        B = det_maps[0].shape[0]

        # ---- parse targets
        boxes_list, labels_list, kpts_list = self._split_targets(batch, B, device)

        # ---- multi-level positive assignment (P3/P4/P5, neighbors, adjacency near boundaries)
        pos_targets = self._build_pos_targets(det_maps, boxes_list, labels_list)

        # ---- detection losses
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

        # ---- keypoint loss (top-down), with optional warmup after freeze
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

    # ---------------------- internals ----------------------

    def _split_targets(
        self,
        batch: Dict[str, Any],
        B: int,
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Accepts either per-image lists (bboxes/labels/kpts) or a single 'targets' tensor."""
        if "bboxes" in batch and "labels" in batch:
            boxes_list = [b.to(device=device, dtype=torch.float32) for b in batch["bboxes"]]
            labels_list = [l.to(device=device, dtype=torch.long) for l in batch["labels"]]
            kpts_raw = batch.get("kpts", [None] * B)
            kpts_list = [
                (k.to(device=device, dtype=torch.float32) if k is not None else torch.empty((0, 2), device=device))
                for k in kpts_raw
            ]
            return boxes_list, labels_list, kpts_list

        t = batch.get("targets", None)
        if t is None or not torch.is_tensor(t):
            empty_b = [torch.zeros((0, 5), device=device) for _ in range(B)]
            empty_l = [torch.zeros((0,), dtype=torch.long, device=device) for _ in range(B)]
            empty_k = [torch.empty((0, 2), device=device) for _ in range(B)]
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
                boxes_list.append(torch.zeros((0, 5), device=device))
                labels_list.append(torch.zeros((0,), dtype=torch.long, device=device))
                kpts_list.append(torch.empty((0, 2), device=device))
                continue
            labels_list.append(tb[:, 1].long())
            boxes_list.append(tb[:, 2:7])  # cx,cy,w,h,ang(rad)
            if has_kp:
                kpts_list.append(tb[:, 7:9])
            else:
                kpts_list.append(torch.empty((0, 2), device=device))
        return boxes_list, labels_list, kpts_list

    def _candidate_levels(self, size_px: float) -> List[int]:
        """Return levels to assign for object of given 'size' (max(w,h) in px)."""
        low, high = self.level_boundaries
        base = 0 if size_px < low else (1 if size_px < high else 2)
        cand = {base}
        # duplicate near boundaries if within margin
        if self.adjacent_level_margin > 0.0:
            if abs(size_px - low) <= self.adjacent_level_margin:
                cand.update({0, 1})
            if abs(size_px - high) <= self.adjacent_level_margin:
                cand.update({1, 2})
        return sorted(cand)

    def _neighbor_offsets(self) -> List[Tuple[int, int]]:
        """Offsets for neighbor positives around (i,j). Radius 1 -> 3x3."""
        R = max(0, self.neighbor_range)
        return [(dj, di) for dj in range(-R, R + 1) for di in range(-R, R + 1)]

    def _build_pos_targets(
        self,
        det_maps: List[torch.Tensor],
        boxes_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
    ) -> List[List[Tuple[int, int, int, Dict[str, float]]]]:
        """For each level, produce positives: (bix, j, i, target dict)."""
        pos_targets: List[List[Tuple[int, int, int, Dict[str, float]]]] = [[] for _ in det_maps]
        neighbor_offs = self._neighbor_offsets()

        for b, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
            if boxes.numel() == 0:
                continue
            cx, cy, w, h, ang = boxes.t()  # radians

            for n in range(boxes.shape[0]):
                size_px = float(max(w[n].item(), h[n].item()))
                cand_levels = self._candidate_levels(size_px)

                for li in cand_levels:
                    stride = self.strides[li]
                    Hf, Wf = det_maps[li].shape[-2], det_maps[li].shape[-1]

                    gx = (cx[n] / stride).item()
                    gy = (cy[n] / stride).item()
                    gi = int(math.floor(gx))
                    gj = int(math.floor(gy))
                    tx = gx - gi
                    ty = gy - gj
                    # NOTE: store targets in *log-space for width/height*
                    tw = float(math.log(max(w[n].item() / stride, 1e-6)))
                    th = float(math.log(max(h[n].item() / stride, 1e-6)))
                    s = float(math.sin(ang[n].item()))
                    c = float(math.cos(ang[n].item()))
                    cls = int(labels[n].item()) if labels.numel() else 0

                    def add_pos(ii: int, jj: int, offx: float, offy: float):
                        if 0 <= ii < Wf and 0 <= jj < Hf:
                            pos_targets[li].append(
                                (b, jj, ii, {"tx": offx, "ty": offy, "tw": tw, "th": th, "sin": s, "cos": c, "cls": cls})
                            )

                    # base
                    add_pos(gi, gj, tx, ty)

                    # neighbors
                    if self.neighbor_cells:
                        for (dj, di) in neighbor_offs:
                            if dj == 0 and di == 0:
                                continue
                            ii = gi + di
                            jj = gj + dj
                            add_pos(ii, jj, gx - ii, gy - jj)

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

            # Split channels: [tx, ty, tw, th, sin, cos, obj, (cls...)]
            tx = dm[:, 0]; ty = dm[:, 1]
            tw = dm[:, 2]; th = dm[:, 3]
            s  = dm[:, 4]; c  = dm[:, 5]
            obj = dm[:, 6]
            cls_logits = dm[:, 7:] if C > 7 else None

            # Objectness targets over entire map
            obj_tgt = torch.zeros_like(obj)

            for (b, j, i, tgt) in pos_targets[li]:
                obj_tgt[b, j, i] = 1.0
                total_pos += 1

                # bbox regression — compare activated preds to targets
                p_tx = torch.sigmoid(tx[b, j, i])
                p_ty = torch.sigmoid(ty[b, j, i])
                # widths/heights are predicted in log-space → exp at decode; 
                # here supervise on decoded scale in *stride units* for stability.
                p_tw = torch.exp(tw[b, j, i])
                p_th = torch.exp(th[b, j, i])

                # --- FIX: compare p_tw to exp(tgt['tw']) and p_th to exp(tgt['th'])
                # (They were mistakenly swapped in the original file.)
                l_box = l_box + self.smoothl1(p_tx, torch.tensor(tgt["tx"], device=device))
                l_box = l_box + self.smoothl1(p_ty, torch.tensor(tgt["ty"], device=device))
                l_box = l_box + self.smoothl1(p_tw, torch.tensor(math.exp(tgt["tw"]), device=device))
                l_box = l_box + self.smoothl1(p_th, torch.tensor(math.exp(tgt["th"]), device=device))

                # angle — normalize pred vector
                p_s = torch.tanh(s[b, j, i])
                p_c = torch.tanh(c[b, j, i])
                v = torch.sqrt(p_s * p_s + p_c * p_c + 1e-9)
                p_sn, p_cn = p_s / v, p_c / v
                l_ang = l_ang + self.smoothl1(p_sn, torch.tensor(tgt["sin"], device=device))
                l_ang = l_ang + self.smoothl1(p_cn, torch.tensor(tgt["cos"], device=device))

                # multi-class (if nc>1)
                if self.num_classes > 1 and cls_logits is not None:
                    y = torch.zeros((self.num_classes,), device=device)
                    y[int(tgt["cls"])] = 1.0
                    l_cls = l_cls + F.binary_cross_entropy_with_logits(cls_logits[b, :, j, i], y)

            # BCE over all cells for objectness
            l_obj = l_obj + F.binary_cross_entropy_with_logits(obj, obj_tgt)

        return l_box, l_obj, l_ang, l_cls, total_pos

    def _keypoint_loss(
        self,
        model: nn.Module,
        feats: List[torch.Tensor],
        boxes_list: List[torch.Tensor],
        kpts_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        """Top-down keypoint loss using GT OBBs to form ROIs on P3 (/8)."""

        if hasattr(model, "module"):  # unwrap DDP
            model = model.module

        device = feats[0].device

        if not hasattr(model, "kpt_from_obbs"):
            return torch.zeros((), device=device), 0

        # Build OBB list (degrees) for ROI
        obb_list: List[torch.Tensor] = []
        total_rois = 0
        for ob in boxes_list:
            if ob is None or ob.numel() == 0:
                obb_list.append(torch.zeros((0, 5), device=device))
                continue
            ob = ob.clone()
            ob[:, 4] = torch.rad2deg(ob[:, 4])
            obb_list.append(ob)
            total_rois += ob.shape[0]

        if total_rois == 0:
            return torch.zeros((), device=device), 0

        # Run top-down head (limit chunk size to prevent OOM in rot_roi)
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

        # Map each meta to its GT keypoint
        for idx, m in enumerate(metas):
            b = int(m.get("b", m.get("bix", 0)))
            oi = int(m.get("oi", idx))  # per-image object index if provided
            if b >= len(kpts_list):
                continue
            kpi = kpts_list[b]
            if kpi is None or kpi.numel() == 0 or oi < 0 or oi >= kpi.shape[0]:
                continue
            M = m["M"]
            uv_px = self._img_kpts_to_crop_uv(kpi[oi:oi+1, :], M, feat_down)  # (1,2) in px
            uv_px = torch.clamp(uv_px, 0, S - 1)
            uv_tgt[idx] = uv_px[0]
            valid[idx] = True

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
        Map image keypoints (x,y) px -> crop (u,v) px using the affine M (2x3) that maps crop->feature.
        Steps: image px -> feature px (divide by feat_down), then invert the 2x3 affine.
        """
        if kpts_img.numel() == 0:
            return torch.empty_like(kpts_img)

        # Ensure device/dtype
        device = M.device
        dtype = M.dtype
        pts = kpts_img.to(device=device, dtype=dtype) / float(feat_down)  # (N,2) in feature px

        # Decompose M: [x_f] = A*u + B*v + C; [y_f] = D*u + E*v + F
        A, B, C = M[0, 0], M[0, 1], M[0, 2]
        D, E, Fv = M[1, 0], M[1, 1], M[1, 2]

        det = A * E - B * D

        # Build inverse of the linear 2x2
        inv_lin = torch.stack([
            torch.stack([ E, -B], dim=0),
            torch.stack([-D,  A], dim=0)
        ], dim=0) / det  # (2,2)

        # Translate points: [x_f - C, y_f - F]
        xy = pts - torch.stack([C, Fv], dim=0)

        # Solve [u, v]^T = inv_lin * [x_f - C, y_f - F]^T
        uv = (inv_lin @ xy.t()).t()  # (N,2)
        return uv
