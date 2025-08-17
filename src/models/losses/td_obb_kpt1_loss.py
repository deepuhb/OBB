# src/models/losses/td_obb_kpt1_loss.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
# Utilities: robust GT extraction for both formats
# -------------------------------------------------------------

def _split_targets_by_image(t: torch.Tensor, B: int, device: torch.device):
    """
    Split YOLO-style 'targets' tensor by image index into per-image lists.
    Expects columns: [img_idx, cls, cx, cy, w, h, angle, (kpx, kpy optional)]
    Returns lists of length B:
      boxes_list:  (Ni,5) -> (cx,cy,w,h,ang)
      labels_list: (Ni,)
      kpts_list:   (Ni,2) or empty (0,2) tensor
    """
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    if t.numel() == 0:
        boxes_list = [torch.zeros((0, 5), dtype=torch.float32, device=device) for _ in range(B)]
        labels_list = [torch.zeros((0,), dtype=torch.long, device=device) for _ in range(B)]
        kpts_list = [torch.empty((0, 2), dtype=torch.float32, device=device) for _ in range(B)]
        return boxes_list, labels_list, kpts_list

    img_idx = t[:, 0].long()
    cls     = t[:, 1].long()
    cx, cy, w, h, ang = t[:, 2], t[:, 3], t[:, 4], t[:, 5], t[:, 6]
    has_kpt = (t.shape[1] >= 9)

    boxes_list, labels_list, kpts_list = [], [], []
    for i in range(B):
        m = (img_idx == i)
        if m.any():
            b = torch.stack([cx[m], cy[m], w[m], h[m], ang[m]], dim=1).to(torch.float32).to(device)
            l = cls[m].to(device)
            if has_kpt:
                k = t[m, 7:9].to(torch.float32).to(device)  # (N,2)
            else:
                k = torch.empty((0, 2), dtype=torch.float32, device=device)
        else:
            b = torch.zeros((0, 5), dtype=torch.float32, device=device)
            l = torch.zeros((0,), dtype=torch.long, device=device)
            k = torch.empty((0, 2), dtype=torch.float32, device=device)
        boxes_list.append(b); labels_list.append(l); kpts_list.append(k)
    return boxes_list, labels_list, kpts_list


def _extract_gt_lists_from_batch(batch: dict, B: int, device: torch.device):
    """
    Robustly extract per-image GT lists from either:
      - lists: batch['bboxes'], batch['labels'], optional batch['kpts']
      - YOLO targets tensor: batch['targets']
    Ensures boxes are (cx,cy,w,h,ang) and keypoints are (N,2) or empty.
    """
    # list format
    if "bboxes" in batch and isinstance(batch["bboxes"], list):
        boxes_list, labels_list, kpts_list = [], [], []
        kpts_in = batch.get("kpts", None)
        for i in range(B):
            bi = torch.as_tensor(batch["bboxes"][i], dtype=torch.float32, device=device)
            if bi.numel() == 0:
                bi = bi.new_zeros((0, 5))
            elif bi.shape[-1] == 4:  # (cx,cy,w,h) -> append ang=0
                zeros = torch.zeros((bi.shape[0], 1), dtype=bi.dtype, device=device)
                bi = torch.cat([bi, zeros], dim=1)
            li = torch.as_tensor(batch.get("labels", [])[i], dtype=torch.long, device=device) \
                 if "labels" in batch else torch.zeros((bi.shape[0],), dtype=torch.long, device=device)
            if kpts_in is not None and i < len(kpts_in) and len(kpts_in[i]):
                ki = torch.as_tensor(kpts_in[i], dtype=torch.float32, device=device)
                if ki.ndim == 3:  # (N,1,2) -> (N,2)
                    ki = ki.squeeze(1)
            else:
                ki = torch.empty((0, 2), dtype=torch.float32, device=device)
            boxes_list.append(bi); labels_list.append(li); kpts_list.append(ki)
        return boxes_list, labels_list, kpts_list

    # YOLO 'targets'
    if "targets" in batch:
        return _split_targets_by_image(batch["targets"], B, device)

    # fallback: empty
    boxes_list = [torch.zeros((0, 5), dtype=torch.float32, device=device) for _ in range(B)]
    labels_list = [torch.zeros((0,), dtype=torch.long, device=device) for _ in range(B)]
    kpts_list = [torch.empty((0, 2), dtype=torch.float32, device=device) for _ in range(B)]
    return boxes_list, labels_list, kpts_list


# -------------------------------------------------------------
# Criterion
# -------------------------------------------------------------

class TDOBBWKpt1Criterion(nn.Module):
    """
    Detection + single-keypoint criterion for YOLO11-style OBB + top-down keypoint.

    Inputs to forward():
      - det_maps: List[Tensor] per level: (B, 7+nc, H, W)
          channel order: [tx, ty, tw, th, sin, cos, obj, cls_0..]
      - feats:    List[Tensor] backbone/FPN features (used by model's kpt head; optional)
      - batch:    dict with either:
                    * lists 'bboxes'/'labels' (and optional 'kpts'), or
                    * YOLO 'targets' tensor [img, cls, cx, cy, w, h, ang, (kpx,kpy)]
      - model:    model handle (optional) for keypoint head calling
      - epoch:    int epoch index (for freeze/warmup)
    Returns:
      total_loss, logs_dict
    """

    def __init__(
        self,
        nc: int,
        strides: Sequence[int] = (8, 16, 32),
        # loss weights
        lambda_box: float = 7.5,
        lambda_obj: float = 3.0,
        lambda_ang: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_kpt: float = 2.0,
        # keypoint training schedule
        kpt_freeze_epochs: int = 0,
        kpt_warmup_epochs: int = 0,
        # routing thresholds (pixels, based on max(w,h))
        level_boundaries: Tuple[float, float] = (32.0, 64.0),  # <=32 -> P3, <=64 -> P4, else P5
        # objectness pos/neg weighting
        obj_pos_weight: float = 1.0,
        obj_neg_weight: float = 1.0,
    ):
        super().__init__()
        self.nc = int(nc)
        self.strides = tuple(int(s) for s in strides)
        assert len(self.strides) >= 1, "Need at least one detection level"
        self.num_levels = len(self.strides)

        # weights
        self.lambda_box = float(lambda_box)
        self.lambda_obj = float(lambda_obj)
        self.lambda_ang = float(lambda_ang)
        self.lambda_cls = float(lambda_cls)
        self.lambda_kpt = float(lambda_kpt)

        # kpt schedule
        self.kpt_freeze_epochs = int(kpt_freeze_epochs)
        self.kpt_warmup_epochs = int(kpt_warmup_epochs)

        # target routing thresholds
        self.level_boundaries = tuple(float(x) for x in level_boundaries)
        if len(self.level_boundaries) != 2:
            raise ValueError("level_boundaries must be a (low, mid) 2-tuple")

        # BCE losses for obj/cls
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    # ---------------------------------------------------------
    # Target building and losses
    # ---------------------------------------------------------

    def _route_level(self, max_side: float) -> int:
        """Return level index based on object size (max side in pixels)."""
        low, mid = self.level_boundaries
        if max_side <= low:  # small -> P3
            return 0
        elif max_side <= mid:  # medium -> P4
            return 1 if self.num_levels >= 2 else 0
        else:  # large -> P5
            return 2 if self.num_levels >= 3 else (1 if self.num_levels >= 2 else 0)

    def _build_targets(
        self,
        det_maps: List[torch.Tensor],
        boxes_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
    ):
        """
        Build per-level targets for anchor-free grids, center-cell assignment.

        Returns:
          targets: list of dicts per level with keys:
            'mask' (B,1,H,W) bool, positives
            'tx','ty','tw','th','sin','cos','obj' (B,1,H,W) float
            'cls' (B,nc,H,W) float (only used if nc>1)
          pos_meta: list of tuples (b, level, gy, gx, cx, cy, w, h, ang)
                    used for keypoint head to sample ROIs if needed
        """
        device = det_maps[0].device
        B = det_maps[0].shape[0]

        # shapes/strides per level
        shapes = [(f.shape[2], f.shape[3]) for f in det_maps]  # (H,W) per level
        s = self.strides

        # allocate targets
        targets = []
        for lvl in range(self.num_levels):
            H, W = shapes[lvl]
            targets.append({
                "mask": torch.zeros((B, 1, H, W), dtype=torch.bool, device=device),
                "tx":   torch.zeros((B, 1, H, W), device=device),
                "ty":   torch.zeros((B, 1, H, W), device=device),
                "tw":   torch.zeros((B, 1, H, W), device=device),
                "th":   torch.zeros((B, 1, H, W), device=device),
                "sin":  torch.zeros((B, 1, H, W), device=device),
                "cos":  torch.zeros((B, 1, H, W), device=device),
                "obj":  torch.zeros((B, 1, H, W), device=device),
                "cls":  torch.zeros((B, self.nc, H, W), device=device) if self.nc > 1 else None,
            })

        pos_meta: List[Tuple[int, int, int, int, float, float, float, float, float]] = []

        # assign GT
        for b in range(B):
            bx = boxes_list[b]  # (N,5)
            if bx.numel() == 0:
                continue
            labs = labels_list[b] if self.nc > 1 else None

            cx, cy, w, h, ang = bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3], bx[:, 4]
            for j in range(bx.shape[0]):
                max_side = float(max(w[j].item(), h[j].item()))
                lvl = self._route_level(max_side)
                H, W = shapes[lvl]
                stride = s[lvl]

                gx = int(torch.clamp((cx[j] / stride).floor(), 0, W - 1).item())
                gy = int(torch.clamp((cy[j] / stride).floor(), 0, H - 1).item())

                T = targets[lvl]
                T["mask"][b, 0, gy, gx] = True
                # offsets in cell space (0..1)
                T["tx"][b, 0, gy, gx] = (cx[j] / stride) - gx
                T["ty"][b, 0, gy, gx] = (cy[j] / stride) - gy
                # log-scale size in stride units
                # clamp to avoid log(0)
                T["tw"][b, 0, gy, gx] = torch.log(torch.clamp(w[j] / stride, min=1e-4))
                T["th"][b, 0, gy, gx] = torch.log(torch.clamp(h[j] / stride, min=1e-4))
                # angle as sin/cos targets
                T["sin"][b, 0, gy, gx] = torch.sin(ang[j])
                T["cos"][b, 0, gy, gx] = torch.cos(ang[j])
                # objectness
                T["obj"][b, 0, gy, gx] = 1.0
                # class if multi-class
                if self.nc > 1 and labs is not None and j < labs.numel():
                    c = int(labs[j].item())
                    if 0 <= c < self.nc:
                        T["cls"][b, c, gy, gx] = 1.0

                pos_meta.append((b, lvl, gy, gx,
                                 float(cx[j].item()), float(cy[j].item()),
                                 float(w[j].item()), float(h[j].item()), float(ang[j].item())))

        return targets, pos_meta

    def _loss_det(self, det_maps: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        """
        Per-level losses for box (tx/ty/tw/th), angle (sin/cos), objectness, and classification.
        We expect det_maps[i] shape: (B, 7+nc, H, W).
        """
        total_box = total_ang = total_obj = total_cls = det_maps[0].new_tensor(0.0)

        for lvl, dm in enumerate(det_maps):
            B, C, H, W = dm.shape
            assert C == (7 + self.nc), f"det_map channels expected 7+nc, got {C}"
            tx, ty = dm[:, 0:1], dm[:, 1:2]
            tw, th = dm[:, 2:3], dm[:, 3:4]
            si, co = dm[:, 4:5], dm[:, 5:6]
            obj    = dm[:, 6:7]
            cls_logit = dm[:, 7:] if self.nc > 1 else None

            T = targets[lvl]
            m = T["mask"]  # (B,1,H,W) bool

            # Box losses (L1): tx/ty use sigmoid to [0,1], tw/th compare raw to log-targets
            if m.any():
                l_tx = F.l1_loss(torch.sigmoid(tx)[m], T["tx"][m], reduction="mean")
                l_ty = F.l1_loss(torch.sigmoid(ty)[m], T["ty"][m], reduction="mean")
                l_tw = F.l1_loss(tw[m], T["tw"][m], reduction="mean")
                l_th = F.l1_loss(th[m], T["th"][m], reduction="mean")
                l_box = l_tx + l_ty + l_tw + l_th

                # Angle loss (L1) on tanh(si/co) vs sin/cos targets
                l_ang = F.l1_loss(torch.tanh(si)[m], T["sin"][m], reduction="mean") + \
                        F.l1_loss(torch.tanh(co)[m], T["cos"][m], reduction="mean")
            else:
                l_box = obj.new_tensor(0.0)
                l_ang = obj.new_tensor(0.0)

            # Objectness BCE on all locations with pos/neg balance
            pos_mask = m
            neg_mask = ~m
            l_obj_pos = self.bce(obj[pos_mask], T["obj"][pos_mask]) if pos_mask.any() else obj.new_tensor(0.0)
            l_obj_neg = self.bce(obj[neg_mask], T["obj"][neg_mask]) if neg_mask.any() else obj.new_tensor(0.0)
            l_obj = self.obj_pos_weight * l_obj_pos + self.obj_neg_weight * l_obj_neg

            # Classification BCE (multi-class one-vs-rest)
            if self.nc > 1 and cls_logit is not None:
                if pos_mask.any():
                    # Only supervise positives for class targets
                    cls_pos_mask = pos_mask.expand_as(cls_logit)
                    l_cls = self.bce(cls_logit[cls_pos_mask], T["cls"][cls_pos_mask])
                else:
                    l_cls = obj.new_tensor(0.0)
            else:
                l_cls = obj.new_tensor(0.0)

            total_box = total_box + l_box
            total_ang = total_ang + l_ang
            total_obj = total_obj + l_obj
            total_cls = total_cls + l_cls

        return total_box, total_ang, total_obj, total_cls

    # ---------------------------------------------------------
    # Keypoint loss hook (optional)
    # ---------------------------------------------------------

    def _predict_kpts_from_feats(
        self,
        model: Optional[nn.Module],
        feats: Optional[List[torch.Tensor]],
        pos_meta: List[Tuple[int, int, int, int, float, float, float, float, float]],
    ) -> Optional[List[torch.Tensor]]:
        """
        Try calling a model-provided keypoint head with (feats, pos_meta).
        Supported method names: forward_kpt, predict_kpt, kpt_head(feats, pos_meta).
        Should return a list of per-image tensors (Ni,2) in absolute pixels.
        """
        if model is None or feats is None:
            return None
        # Try a few common entry points
        if hasattr(model, "forward_kpt"):
            return model.forward_kpt(feats, pos_meta)
        if hasattr(model, "predict_kpt"):
            return model.predict_kpt(feats, pos_meta)
        if hasattr(model, "kpt_head"):
            khead = model.kpt_head
            if callable(khead):
                return khead(feats, pos_meta)
            if hasattr(khead, "forward"):
                return khead.forward(feats, pos_meta)
        return None

    def _loss_kpt(
        self,
        model: Optional[nn.Module],
        feats: Optional[List[torch.Tensor]],
        pos_meta: List[Tuple[int, int, int, int, float, float, float, float, float]],
        kpts_list: List[torch.Tensor],   # per-image (Ni,2) targets in absolute pixels
    ):
        """
        Compute L1 loss between predicted kpts and GT for positives (if model head is available).
        Returns (loss, n_pos).
        """
        device = feats[0].device if (feats and len(feats)) else (kpts_list[0].device if len(kpts_list) else torch.device("cpu"))
        preds = self._predict_kpts_from_feats(model, feats, pos_meta)
        if preds is None:
            return torch.zeros((), device=device), 0

        # preds is list per image (Ni,2). kpts_list is same shape per image.
        total = torch.zeros((), device=device)
        npos = 0
        # Reconstruct per-image grouping from pos_meta
        # pos_meta entries are appended in order; we count how many per image
        per_image_counts: Dict[int, int] = {}
        for b, *_ in pos_meta:
            per_image_counts[b] = per_image_counts.get(b, 0) + 1

        # Iterate images by index and compare
        for i, (gt_k, pred_k) in enumerate(zip(kpts_list, preds)):
            if gt_k.numel() == 0 or pred_k is None or pred_k.numel() == 0:
                continue
            # If counts mismatch, align by min
            n = min(gt_k.shape[0], pred_k.shape[0])
            if n <= 0:
                continue
            total = total + F.l1_loss(pred_k[:n], gt_k[:n], reduction="mean")
            npos += n

        if npos == 0:
            return torch.zeros((), device=device), 0
        return total, npos

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    def forward(
        self,
        det_maps: List[torch.Tensor] | torch.Tensor,
        feats: Optional[List[torch.Tensor]],
        batch: Dict[str, Any],
        model: Optional[nn.Module] = None,
        epoch: Optional[int] = None
    ):
        """
        Compute total loss and a dict of logs.
        """
        # ensure list of levels
        if isinstance(det_maps, torch.Tensor):
            det_maps = [det_maps]
        device = det_maps[0].device
        B = batch["image"].shape[0] if isinstance(batch.get("image", None), torch.Tensor) else len(batch.get("bboxes", []))

        # Extract GT lists (robust to presence/absence of 'kpts')
        boxes_list, labels_list, kpts_list = _extract_gt_lists_from_batch(batch, B, device)

        # Build multi-scale targets
        targets, pos_meta = self._build_targets(det_maps, boxes_list, labels_list)

        # Compute detection losses
        l_box, l_ang, l_obj, l_cls = self._loss_det(det_maps, targets)

        # Compute keypoint loss (optional)
        l_kpt, kpt_pos = self._loss_kpt(model, feats, pos_meta, kpts_list)

        # Keypoint freeze/warmup schedule
        kpt_scale = 0.0
        if self.lambda_kpt > 0.0:
            ep = int(epoch) if epoch is not None else 0
            if ep < self.kpt_freeze_epochs:
                kpt_scale = 0.0
            else:
                if self.kpt_warmup_epochs > 0:
                    # linearly ramp from 0->1 over warmup epochs
                    t = min(1.0, (ep - self.kpt_freeze_epochs + 1) / float(self.kpt_warmup_epochs))
                    kpt_scale = t
                else:
                    kpt_scale = 1.0

        # Weighted sum
        total = (
            self.lambda_box * l_box +
            self.lambda_ang * l_ang +
            self.lambda_obj * l_obj +
            self.lambda_cls * l_cls +
            self.lambda_kpt * kpt_scale * l_kpt
        )

        # Logs
        logs = {
            "box_loss": float(l_box.detach().item()),
            "obj_loss": float(l_obj.detach().item()),
            "ang_loss": float(l_ang.detach().item()),
            "kpt_loss": float((kpt_scale * l_kpt).detach().item()) if isinstance(l_kpt, torch.Tensor) else 0.0,
            "kc_loss": 0.0,  # reserved for kpt classification if you add it later
            "Pos": float(sum(int(t["mask"].sum().item()) for t in targets) / max(1, B)),
        }
        return total, logs
