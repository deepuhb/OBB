# src/models/losses/td_obb_kpt1_loss.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from typing import Any, Dict, List, Tuple
import math
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------
# Utilities: robust GT extraction for both formats
# -------------------------------------------------------------
def _is_dist():
    return dist.is_available() and dist.is_initialized()

@torch.no_grad()
def _synth_center_obbs(batch_size, img_h, img_w, device, deg=0.0):
    # 1 ROI per image, centered, 1/3 of min side
    S = min(img_h, img_w)
    w = h = float(S) / 3.0
    cx = float(img_w) / 2.0
    cy = float(img_h) / 2.0
    obb = torch.tensor([cx, cy, w, h, deg], device=device).view(1, 5)
    return [obb.clone() for _ in range(batch_size)]


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

def _to_list_of_tensors(seq, device, dtype):
    out = []
    for x in seq:
        if x is None:
            out.append(torch.zeros(0, dtype=dtype, device=device))
        elif torch.is_tensor(x):
            out.append(x.to(device=device, dtype=dtype))
        else:
            out.append(torch.as_tensor(x, device=device, dtype=dtype))
    return out

def _extract_gt_lists_from_batch(batch, B, device):
    """
    Prefer dataset-prepared keys:
      - 'bboxes' : (Ni,5) [cx,cy,w,h,theta_rad] in pixels
      - 'labels' : (Ni,)
      - 'kpts'   : (Ni,2) in pixels
    Fallback to 'targets' only if needed.
    """
    if "bboxes" in batch and "labels" in batch and "kpts" in batch:
        boxes_list  = _to_list_of_tensors(batch["bboxes"], device, torch.float32)
        labels_list = _to_list_of_tensors(batch["labels"], device, torch.long)
        kpts_list   = _to_list_of_tensors(batch["kpts"],   device, torch.float32)
        assert len(boxes_list) == B and len(labels_list) == B and len(kpts_list) == B, \
            "GT lists must match batch size"
        return boxes_list, labels_list, kpts_list

    # Fallback: parse 'targets' here if you still want to support it
    if "targets" in batch and batch["targets"] is not None:
        raise RuntimeError("Loss fell back to 'targets'. Either provide bboxes/labels/kpts or implement this branch.")

    # No GT available
    z5 = torch.zeros(0, 5, device=device)
    zc = torch.zeros(0,    device=device, dtype=torch.long)
    zk = torch.zeros(0, 2, device=device)
    return [z5]*B, [zc]*B, [zk]*B

def _invert_affine_2x3(M: torch.Tensor) -> torch.Tensor:
    # M: (2,3) mapping [u,v,1] (crop px) -> [x_feat,y_feat] (feature px)
    # Return Minv: (2,3) mapping [x_feat,y_feat,1] -> [u,v] (crop px)
    A = M[:, :2]   # (2,2)
    t = M[:, 2:]   # (2,1)
    det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    if abs(float(det)) < 1e-12:
        # degenerate; fall back to identity on SxS center
        I = torch.tensor([[1.,0.,0.],[0.,1.,0.]], device=M.device, dtype=M.dtype)
        return I
    invA00 =  A[1,1]/det
    invA01 = -A[0,1]/det
    invA10 = -A[1,0]/det
    invA11 =  A[0,0]/det
    invA = torch.stack([torch.stack([invA00, invA01]),
                        torch.stack([invA10, invA11])])
    invt = -invA @ t
    return torch.cat([invA, invt], dim=1)  # (2,3)

def _img_kpts_to_crop_uv(kpt_xy_img: torch.Tensor,
                         M_crop_to_feat: torch.Tensor,
                         feat_down: float) -> torch.Tensor:
    # kpt_xy_img: (N,2) in image px; M maps crop px → feature px
    # Convert image px → feature px, then apply inverse M to get crop px (u,v)
    xy_feat = kpt_xy_img / float(feat_down)
    Minv = _invert_affine_2x3(M_crop_to_feat)
    # [u,v]^T = Minv[:,:2] @ [x,y]^T + Minv[:,2]
    uv = (Minv[:, :2] @ xy_feat.T + Minv[:, 2:]).T  # (N,2)
    return uv


class TDOBBWKpt1Criterion(nn.Module):
    """
    Detection + single-keypoint criterion for YOLO11-style OBB + top-down keypoint.

    det_maps: List of detection maps per level, each (B, 7+nc, H, W)
              channel order: [tx, ty, tw, th, sin, cos, obj, cls_0..]
    feats   : backbone/FPN features (for keypoint head; optional)
    batch   : dict with either lists (bboxes/labels[/kpts]) or YOLO 'targets'
    model   : optional model handle for keypoint head
    epoch   : int (for kpt freeze/warmup)
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
        level_boundaries: Tuple[float, float] = (32.0, 64.0),  # <=32 -> P3, <=64 -> P4, else -> P5
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

        # routing thresholds
        self.level_boundaries = tuple(float(x) for x in level_boundaries)
        if len(self.level_boundaries) != 2:
            raise ValueError("level_boundaries must be a (low, mid) 2-tuple")

        # BCE losses for obj/cls
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

        # <<< IMPORTANT: define obj weights so _loss_det can use them >>>
        self.obj_pos_weight = float(obj_pos_weight)
        self.obj_neg_weight = float(obj_neg_weight)

    # ---------------------------------------------------------
    # Target building and losses
    # ---------------------------------------------------------

    def _route_level(self, max_side: float) -> int:
        """Return level index based on object size (max side in pixels)."""
        low, mid = self.level_boundaries
        if max_side <= low:      # small -> P3
            return 0
        elif max_side <= mid:    # medium -> P4
            return 1 if self.num_levels >= 2 else 0
        else:                    # large -> P5
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

        shapes = [(f.shape[2], f.shape[3]) for f in det_maps]  # (H,W)
        s = self.strides

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
                T["tx"][b, 0, gy, gx] = (cx[j] / stride) - gx
                T["ty"][b, 0, gy, gx] = (cy[j] / stride) - gy
                T["tw"][b, 0, gy, gx] = torch.log(torch.clamp(w[j] / stride, min=1e-4))
                T["th"][b, 0, gy, gx] = torch.log(torch.clamp(h[j] / stride, min=1e-4))
                T["sin"][b, 0, gy, gx] = torch.sin(ang[j])
                T["cos"][b, 0, gy, gx] = torch.cos(ang[j])
                T["obj"][b, 0, gy, gx] = 1.0
                if self.nc > 1 and labs is not None and j < labs.numel():
                    c = int(labs[j].item())
                    if 0 <= c < self.nc:
                        T["cls"][b, c, gy, gx] = 1.0

                deg = float(torch.rad2deg(ang[j]).item())
                pos_meta.append((
                    b, lvl, gy, gx,
                    float(cx[j].item()),
                    float(cy[j].item()),
                    float(w[j].item()),
                    float(h[j].item()),
                    deg))

        return targets, pos_meta

    def _loss_det(self, det_maps: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        """
        Per-level losses for box (tx/ty/tw/th), angle (sin/cos), objectness, and classification.
        det_maps[i] shape: (B, 7+nc, H, W).
        """
        total_box = total_ang = total_obj = total_cls = det_maps[0].new_tensor(0.0)

        # Safety fallback if class attrs missing for any reason
        obj_pos_w = getattr(self, "obj_pos_weight", 1.0)
        obj_neg_w = getattr(self, "obj_neg_weight", 1.0)

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

            if m.any():
                l_tx = F.l1_loss(torch.sigmoid(tx)[m], T["tx"][m], reduction="mean")
                l_ty = F.l1_loss(torch.sigmoid(ty)[m], T["ty"][m], reduction="mean")
                l_tw = F.l1_loss(tw[m], T["tw"][m], reduction="mean")
                l_th = F.l1_loss(th[m], T["th"][m], reduction="mean")
                l_box = l_tx + l_ty + l_tw + l_th

                l_ang = F.l1_loss(torch.tanh(si)[m], T["sin"][m], reduction="mean") + \
                        F.l1_loss(torch.tanh(co)[m], T["cos"][m], reduction="mean")
            else:
                z = obj.new_tensor(0.0)
                l_box = z; l_ang = z

            # Objectness on pos and neg
            pos_mask = m
            neg_mask = ~m
            l_obj_pos = self.bce(obj[pos_mask], T["obj"][pos_mask]) if pos_mask.any() else obj.new_tensor(0.0)
            l_obj_neg = self.bce(obj[neg_mask], T["obj"][neg_mask]) if neg_mask.any() else obj.new_tensor(0.0)
            l_obj = obj_pos_w * l_obj_pos + obj_neg_w * l_obj_neg

            # Classification (positives only)
            if self.nc > 1 and cls_logit is not None:
                if pos_mask.any():
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


    def _predict_kpts_from_feats(self, model, feats, pos_meta):
        """
        Use model helper (ROI + head). Compatible with both (preds, metas)
        and preds-only returns.
        """
        out = model.kpt_from_obbs(feats, pos_meta)  # evaluator also calls with kwargs
        if isinstance(out, tuple):
            preds, _ = out
            return preds
        return out

    def _loss_kpt(self, model, feats, pos_meta, kpts_list):
        """
        Keypoint loss with:
          - det-ROI first, GT-OBB fallback (angles in DEG),
          - DDP-safe dummy forward when a rank has 0 valid samples but others don't.
        Returns:
          l_kpt (scalar tensor), valid_count (int)
        """
        # --- basic config ---
        feat0 = feats[0] if isinstance(feats, (list, tuple)) else feats
        device = feat0.device
        B, _, Hf, Wf = feat0.shape

        roi = getattr(model, "roi", None)
        S = getattr(roi, "S", 64)
        feat_down = getattr(roi, "feat_down", 8)
        kpt_weight = getattr(self, "lambda_kpt", 1.0)

        # --- 1) try detection-based ROIs ---
        uv_pred, metas = model.kpt_from_obbs(feats, pos_meta)  # (N,2), [{'bix', 'M'}, ...]
        n_rois = int(uv_pred.shape[0]) if torch.is_tensor(uv_pred) else 0

        # --- 2) GT fallback (image px, angle in DEG) ---
        if n_rois == 0:
            # boxes source prepared earlier in forward()
            boxes_src = getattr(self, "_boxes_list", None)
            if boxes_src is None:
                boxes_src = [None] * len(kpts_list)

            obb_list = []
            for boxes_b, _ in zip(boxes_src, kpts_list):
                if boxes_b is None or (torch.is_tensor(boxes_b) and boxes_b.numel() == 0):
                    obb_list.append(torch.zeros((0, 5), device=device))
                else:
                    ob = boxes_b.to(device).clone()  # (Ni,5) cx,cy,w,h,theta(rad)
                    ob[:, 4] = ob[:, 4] * (180.0 / math.pi)  # rad -> deg
                    obb_list.append(ob)

            uv_pred, metas = model.kpt_from_obbs(feats, obb_list)
            n_rois = int(uv_pred.shape[0]) if torch.is_tensor(uv_pred) else 0

        # --- If still nothing, maybe other ranks have samples -> dummy head forward here ---
        if n_rois == 0:
            # Check if ANY rank has kpt samples this step
            has_local = torch.tensor([0], device=device, dtype=torch.int32)
            any_has = has_local.clone()
            if _is_dist():
                dist.all_reduce(any_has, op=dist.ReduceOp.MAX)

            l_kpt = torch.zeros((), device=device)
            if any_has.item() == 1:
                # Build one synthetic ROI per image around center of IMAGE (in px, deg)
                # Need image resolution: infer from feature map & stride
                img_h = Hf * int(feat_down)
                img_w = Wf * int(feat_down)
                synth_obbs = _synth_center_obbs(B, img_h, img_w, device, deg=0.0)
                # ROI is @no_grad(), so crops are constants; the head still gets a graph via its own weights.
                uv_dummy, _ = model.kpt_from_obbs(feats, synth_obbs)  # (B,2) or (N,2)
                if torch.is_tensor(uv_dummy) and uv_dummy.numel() > 0:
                    l_kpt = l_kpt + uv_dummy.sum() * 0.0  # zero-cost dummy that uses kpt head params
            return l_kpt, 0

        # --- 3) Build GT uv targets per ROI using metas['M'] mapping (crop px -> feat px) ---
        uv_tgt = torch.zeros_like(uv_pred, device=device)
        valid = torch.zeros((n_rois,), dtype=torch.bool, device=device)

        idx = 0
        for m in metas:
            # metas are detached in roi; still convert dtype/device explicitly
            b = int(m.get('b', m.get('bix', 0)))
            M = m['M'].to(device=device, dtype=uv_pred.dtype)

            if b >= len(kpts_list) or kpts_list[b] is None or kpts_list[b].numel() == 0:
                idx += 1
                continue

            k_b = kpts_list[b]  # (Ni,2) in IMAGE px
            uv_all = _img_kpts_to_crop_uv(k_b, M, feat_down)  # -> (Ni,2) crop px
            uv_all = torch.clamp(uv_all, 0, S - 1)

            d2 = (uv_all[:, 0] - (S / 2)) ** 2 + (uv_all[:, 1] - (S / 2)) ** 2
            j = int(torch.argmin(d2).item())
            uv_tgt[idx] = uv_all[j]
            valid[idx] = True
            idx += 1

        vcnt = int(valid.sum().item())

        # If no valid mapping on this rank but others have ROIs, run dummy forward for DDP
        if vcnt == 0:
            has_local = torch.tensor([1], device=device, dtype=torch.int32)  # we *had* ROIs, mapping failed locally
            any_has = has_local.clone()
            if _is_dist():
                dist.all_reduce(any_has, op=dist.ReduceOp.MAX)

            l_kpt = torch.zeros((), device=device)
            if any_has.item() == 1:
                img_h = Hf * int(feat_down)
                img_w = Wf * int(feat_down)
                synth_obbs = _synth_center_obbs(B, img_h, img_w, device, deg=0.0)
                uv_dummy, _ = model.kpt_from_obbs(feats, synth_obbs)
                if torch.is_tensor(uv_dummy) and uv_dummy.numel() > 0:
                    l_kpt = l_kpt + uv_dummy.sum() * 0.0
            return l_kpt, 0

        # --- 4) real loss ---
        uv_pred_v = uv_pred[valid]
        uv_tgt_v = uv_tgt[valid]
        l_kpt = F.smooth_l1_loss(uv_pred_v, uv_tgt_v, reduction='mean')

        # Print once (rank 0)
        if (not _is_dist()) or dist.get_rank() == 0:
            if not hasattr(self, "_kpt_dbg_once"):
                self._kpt_dbg_once = True
                print(
                    f"[kpt] ROIs {n_rois}, valid {vcnt}, l_kpt_raw {float(l_kpt.detach().item()):.4f}, weight {kpt_weight}")

        return l_kpt, vcnt

    def forward(
        self,
        det_maps: List[torch.Tensor] | torch.Tensor,
        feats: Optional[List[torch.Tensor]],
        batch: Dict[str, Any],
        model: Optional[nn.Module] = None,
        epoch: Optional[int] = None
    ):
        """Compute total loss and a dict of logs."""
        # ensure list of levels
        if isinstance(det_maps, torch.Tensor):
            det_maps = [det_maps]
        device = det_maps[0].device

        # batch size
        if isinstance(batch.get("image", None), torch.Tensor):
            B = batch["image"].shape[0]
        else:
            B = len(batch.get("bboxes", [])) if "bboxes" in batch else int(batch.get("batch_size", 0) or 0)

        # Extract GT lists (robust to presence/absence of 'kpts')
        boxes_list, labels_list, kpts_list = _extract_gt_lists_from_batch(batch, B, device)
        self._boxes_list = boxes_list  # for GT fallback in _loss_kpt

        # Build multi-scale targets
        targets, pos_meta = self._build_targets(det_maps, boxes_list, labels_list)

        # --- count positives across all levels ---
        pos_total = 0
        for T in targets:  # each T has T["mask"] : (B,1,H,W) bool
            pos_total += int(T["mask"].sum().item())

        # batch size
        if isinstance(batch.get("image", None), torch.Tensor):
            B = int(batch["image"].shape[0])
        else:
            B = len(boxes_list)

        pos_per_img = pos_total / max(B, 1)

        # Detection losses
        l_box, l_ang, l_obj, l_cls = self._loss_det(det_maps, targets)

        # Keypoint loss (optional)
        l_kpt, kpt_pos = self._loss_kpt(model, feats, pos_meta, kpts_list)

        # Keypoint freeze/warmup schedule
        kpt_scale = 0.0
        if self.lambda_kpt > 0.0:
            ep = int(epoch) if epoch is not None else 0
            if ep < self.kpt_freeze_epochs:
                kpt_scale = 0.0
            else:
                if self.kpt_warmup_epochs > 0:
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
            "loss_box": float(l_box.detach().item()),
            "loss_obj": float(l_obj.detach().item()),
            "loss_ang": float(l_ang.detach().item()),
            "loss_kpt": float((kpt_scale * l_kpt).detach().item()),
            "loss_kc": 0.0,
            "loss_cls": float(l_cls.detach().item()) if isinstance(l_cls, torch.Tensor) else float(l_cls),
            "num_pos": float(pos_per_img),

            "kpt_loss_raw": l_kpt * kpt_scale,
        }

        return total, logs
