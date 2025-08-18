# src/models/losses/td_obb_kpt1_loss.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from typing import Any, Dict, List, Tuple
import math
import numpy as np

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

def _extract_gt_lists_from_batch(batch: Dict[str, Any], B: int, device: torch.device):
    """
    Returns per-image GT (length B):
      boxes_list[i]  : (Ni,5)  (cx,cy,w,h,ang_rad) in PIXELS
      labels_list[i] : (Ni,)
      kpts_list[i]   : (Ni,2)  in PIXELS (1 keypoint supported here)

    Priority:
      1) use batch['bboxes']/['labels']/['kpts'] if they contain any instances
      2) else parse batch['targets'] in multiple tolerated formats
    """

    def empty_boxes():  return torch.zeros((0,5), dtype=torch.float32, device=device)
    def empty_labels(): return torch.zeros((0,),  dtype=torch.long,   device=device)
    def empty_kpts():   return torch.zeros((0,2), dtype=torch.float32, device=device)

    # Image size for de-normalization
    H = W = None
    imgs = batch.get('image', None)
    if torch.is_tensor(imgs) and imgs.ndim >= 4:
        H, W = int(imgs.shape[-2]), int(imgs.shape[-1])
    elif isinstance(imgs, list) and imgs and torch.is_tensor(imgs[0]):
        H, W = int(imgs[0].shape[-2]), int(imgs[0].shape[-1])

    boxes_list  = [empty_boxes()  for _ in range(B)]
    labels_list = [empty_labels() for _ in range(B)]
    kpts_list   = [empty_kpts()   for _ in range(B)]

    # ---------- 1) prefer per-image lists if they have any instances ----------
    boxes_in  = batch.get('bboxes', None)
    labels_in = batch.get('labels', None)
    kpts_in   = batch.get('kpts',   None)

    def total_instances(lst) -> int:
        if not isinstance(lst, (list, tuple)): return 0
        tot = 0
        for x in lst:
            if torch.is_tensor(x):         tot += int(x.shape[0]) if x.ndim >= 2 else 0
            elif isinstance(x, (list,tuple)): tot += len(x)
        return tot

    use_per_image = (
        isinstance(boxes_in,  (list,tuple)) and
        isinstance(labels_in, (list,tuple)) and
        total_instances(boxes_in) > 0
    )

    if use_per_image:
        for i in range(B):
            bx = boxes_in[i] if i < len(boxes_in) else None
            lb = labels_in[i] if i < len(labels_in) else None
            kp = (kpts_in[i] if isinstance(kpts_in,(list,tuple)) and i < len(kpts_in) else None)

            if torch.is_tensor(bx) and bx.numel():
                bx = bx.to(device=device, dtype=torch.float32).reshape(-1, bx.shape[-1])
                if   bx.shape[-1] > 5: bx = bx[:, :5]
                elif bx.shape[-1] < 5:
                    z = torch.zeros((bx.size(0),5), dtype=torch.float32, device=device)
                    z[:, :bx.shape[1]] = bx
                    bx = z
                boxes_list[i] = bx
            if torch.is_tensor(lb) and lb.numel():
                labels_list[i] = lb.to(device=device, dtype=torch.long).reshape(-1)
            if torch.is_tensor(kp) and kp.numel():
                kp = kp.to(device=device, dtype=torch.float32).reshape(-1, kp.shape[-1])
                kpts_list[i] = kp[:, :2] if kp.shape[-1] >= 2 else kpts_list[i]
        return boxes_list, labels_list, kpts_list

    # ---------- 2) robust fallback: parse batch['targets'] ----------
    targets = batch.get('targets', None)
    if targets is None:
        return boxes_list, labels_list, kpts_list

    # helpers
    def denorm_xy(x, y):
        if H is None or W is None:
            return float(x), float(y)
        return float(x) * W, float(y) * H

    def poly4_to_obb_deg(pts_px: np.ndarray):
        (x1,y1),(x2,y2),(x3,y3),(x4,y4) = pts_px.astype(np.float32)
        cx = (x1 + x2 + x3 + x4) * 0.25
        cy = (y1 + y2 + y3 + y4) * 0.25
        w  = float(math.hypot(x2 - x1, y2 - y1))
        h  = float(math.hypot(x3 - x2, y3 - y2))
        ang_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
        return cx, cy, max(w,1.0), max(h,1.0), ang_deg

    def parse_rows_tensor(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Accept a tensor of shape (N,M).
        Supported M:
          11: cls + 8 poly + 2 kpt (normalized)
          12: img_idx + cls + 8 poly + 2 kpt (normalized)  -> img_idx handled outside
          6/7: cls + cx,cy,w,h,(ang)  (cxcywh normalized if <=1.5 else pixels; ang rad if |ang|<=~pi*1.5 else deg)
        Returns boxes(N,5 rad pixels), labels(N,), kpts(N,2 pixels)
        """
        t = t.to(device=device, dtype=torch.float32).reshape(-1, t.shape[-1])
        M = t.shape[-1]
        if M == 11:
            cls = t[:, 0].to(torch.long)
            poly = t[:, 1:9].view(-1, 4, 2)
            kxy  = t[:, 9:11]
            obbs, kpts = [], []
            for P, K in zip(poly, kxy):
                pts_px = np.array([denorm_xy(P[i,0].item(), P[i,1].item()) for i in range(4)], dtype=np.float32)
                cx,cy,w,h,ang_deg = poly4_to_obb_deg(pts_px)
                obbs.append([cx,cy,w,h, math.radians(ang_deg)])
                kx,ky = denorm_xy(K[0].item(), K[1].item())
                kpts.append([kx,ky])
            return (torch.tensor(obbs, dtype=torch.float32, device=device),
                    cls.reshape(-1),
                    torch.tensor(kpts, dtype=torch.float32, device=device))
        elif M == 6 or M == 7:
            # cls, cx,cy,w,h,(ang). Center/size likely normalized; detect.
            cls = t[:, 0].to(torch.long)
            cx, cy, w, h = t[:, 1], t[:, 2], t[:, 3], t[:, 4]
            # heuristic: if values look normalized, denorm them
            if torch.max(torch.stack([cx.abs(), cy.abs(), w.abs(), h.abs()])) <= 1.5 and H and W:
                cx, cy = cx * W, cy * H
                w,  h  = w * W,  h * H
            if M == 7:
                ang = t[:, 5]
                # detect deg vs rad
                ang = torch.where(ang.abs() > math.pi * 1.5, torch.deg2rad(ang), ang)
            else:
                ang = torch.zeros_like(cx)
            obb = torch.stack([cx, cy, w.clamp_min(1.0), h.clamp_min(1.0), ang], dim=1)
            # no kpt provided -> default to box center (ok for loss gating)
            kpt = torch.stack([cx, cy], dim=1)
            return obb, cls.reshape(-1), kpt
        else:
            # Unsupported width
            return empty_boxes(), empty_labels(), empty_kpts()

    def parse_maybe_line(line: Union[str, List[float], np.ndarray, torch.Tensor]) -> torch.Tensor:
        if torch.is_tensor(line):
            return line
        if isinstance(line, (list, tuple, np.ndarray)):
            arr = np.asarray(line, dtype=np.float32).reshape(1, -1)
            return torch.tensor(arr, dtype=torch.float32, device=device)
        if isinstance(line, str):
            vals = [float(v) for v in line.strip().split()]
            arr = np.asarray(vals, dtype=np.float32).reshape(1, -1)
            return torch.tensor(arr, dtype=torch.float32, device=device)
        return torch.zeros((0,11), dtype=torch.float32, device=device)

    # targets can be per-image list OR single big tensor (with img_idx)
    if isinstance(targets, (list, tuple)):
        for i in range(min(B, len(targets))):
            ti = targets[i]
            if ti is None:
                continue
            if isinstance(ti, (list, tuple)) and ti and not torch.is_tensor(ti):
                # list of rows/lines
                rows = [parse_maybe_line(r) for r in ti]
                if len(rows):
                    T = torch.cat(rows, dim=0)
                else:
                    T = torch.zeros((0,11), dtype=torch.float32, device=device)
            else:
                T = parse_maybe_line(ti)
            if T.numel() == 0:
                continue
            if T.size(1) == 12:
                # drop img_idx column for per-image path
                T = T[:, 1:]
            bx, lb, kp = parse_rows_tensor(T)
            boxes_list[i], labels_list[i], kpts_list[i] = bx, lb, kp
        return boxes_list, labels_list, kpts_list

    if torch.is_tensor(targets):
        T = targets.to(device=device, dtype=torch.float32)
        if T.ndim == 2 and T.size(1) >= 12:
            # img_idx + rest (normalized)
            bix = T[:, 0].to(torch.long).clamp_(0, B-1)
            rows = T[:, 1:]
            for i in range(B):
                sel = (bix == i)
                if sel.any():
                    bx, lb, kp = parse_rows_tensor(rows[sel])
                    boxes_list[i], labels_list[i], kpts_list[i] = bx, lb, kp
            return boxes_list, labels_list, kpts_list
        elif T.ndim == 2 and (T.size(1) in (11,6,7)):
            bx, lb, kp = parse_rows_tensor(T)
            boxes_list[0], labels_list[0], kpts_list[0] = bx, lb, kp
            return boxes_list, labels_list, kpts_list

    # final fallback: still nothing -> one-time concise debug
    if not hasattr(_extract_gt_lists_from_batch, "_dbg_once"):
        _extract_gt_lists_from_batch._dbg_once = True
        try:
            t0 = targets[0] if isinstance(targets,(list,tuple)) and len(targets) else targets
            tinfo = f"type={type(t0)}"
            if torch.is_tensor(t0): tinfo += f" shape={tuple(t0.shape)}"
        except Exception:
            tinfo = "uninspectable targets"
        print(f"[extractor] could not parse targets; first item {tinfo}. "
              f"Expected: 11 cols (cls+8poly+2kpt) or 12 (img_idx+...).")
    return boxes_list, labels_list, kpts_list



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

    # ---------------------------------------------------------
    # Keypoint loss hook (optional)
    # ---------------------------------------------------------

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

    def _loss_kpt(
        self,
        model: Optional[nn.Module],
        feats: Optional[List[torch.Tensor]],
        pos_meta: List[Tuple[int, int, int, int, float, float, float, float, float]],
        kpts_list: List[torch.Tensor],   # per-image (Ni,2) targets in absolute pixels
    ):
        """
        Compute L1 loss between predicted kpts and GT for positives (if model head available).
        Returns (loss, n_pos).
        """
        device = feats[0].device if (feats and len(feats)) else \
                 (kpts_list[0].device if len(kpts_list) else torch.device("cpu"))
        preds = self._predict_kpts_from_feats(model, feats, pos_meta)
        if preds is None:
            return torch.zeros((), device=device), 0

        total = torch.zeros((), device=device)
        npos = 0

        # Group by image index (pos_meta is appended in order)
        # We simply compare per-image in given order; if counts mismatch, align by min(N_gt, N_pred)
        for i, (gt_k, pred_k) in enumerate(zip(kpts_list, preds)):
            if gt_k.numel() == 0 or pred_k is None or pred_k.numel() == 0:
                continue
            n = min(gt_k.shape[0], pred_k.shape[0])
            if n <= 0:
                continue
            total = total + F.l1_loss(pred_k[:n], gt_k[:n], reduction="mean")
            npos += n

        if npos == 0:
            return torch.zeros((), device=device), 0
        return total, npos

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

        # sanity check-----
        gt_total = sum(b.size(0) for b in boxes_list)
        if epoch == 0 and gt_total == 0:
            print("[loss] no GT after extractor — check 'targets' format (expect 11 cols: cls + 8 poly + 2 kpt)")



        # Build multi-scale targets
        targets, pos_meta = self._build_targets(det_maps, boxes_list, labels_list)

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
            "box_loss": float(l_box.detach().item()),
            "obj_loss": float(l_obj.detach().item()),
            "ang_loss": float(l_ang.detach().item()),
            "kpt_loss": float((kpt_scale * l_kpt).detach().item()),
            "kc_loss": 0.0,
            "Pos": ...,
            "box_loss_raw": l_box,
            "obj_loss_raw": l_obj,
            "ang_loss_raw": l_ang,
            "kpt_loss_raw": l_kpt * kpt_scale,
            "kc_loss_raw": torch.tensor(0.0, device=device),
            }
        return total, logs
