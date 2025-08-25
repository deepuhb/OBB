# src/models/losses/td_obb_kpt1_loss.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

# ------------------------- small utils -------------------------

def _dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()

def _ddp_mean(x: torch.Tensor) -> torch.Tensor:
    """
    All-reduce mean for floating tensors. Always computes in float to avoid
    int/float casting errors across ranks.
    """
    y = x.detach().clone()
    if y.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        y = y.to(torch.float32)
    if _dist_ready():
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        y = y / float(dist.get_world_size())
    return y

def _ddp_sum(x: torch.Tensor) -> torch.Tensor:
    """
    All-reduce sum for counters (int or float). Keeps dtype.
    """
    y = x.detach().clone()
    if _dist_ready():
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y

def _invert_affine_2x3(M: torch.Tensor) -> torch.Tensor:
    """Invert a 2x3 affine (maps crop px -> feat px) to (feat px -> crop px)."""
    A = M[:, :2]  # (2,2)
    t = M[:, 2:]  # (2,1)
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(float(det)) < 1e-12:
        I = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=M.device, dtype=M.dtype)
        return I
    invA00 =  A[1, 1] / det
    invA01 = -A[0, 1] / det
    invA10 = -A[1, 0] / det
    invA11 =  A[0, 0] / det
    invA = torch.stack([torch.stack([invA00, invA01]),
                        torch.stack([invA10, invA11])])
    invt = -invA @ t
    return torch.cat([invA, invt], dim=1)  # (2,3)

def _img_kpts_to_crop_uv(kpt_xy_img: torch.Tensor,
                         M_crop_to_feat: torch.Tensor,
                         feat_down: float) -> torch.Tensor:
    """
    Convert GT keypoints from image px to crop uv (in crop px).
    We: image px -> feature px (divide by feat_down) -> apply inverse(M).
    """
    xy_feat = kpt_xy_img / float(feat_down)
    Minv = _invert_affine_2x3(M_crop_to_feat)
    uv = (Minv[:, :2] @ xy_feat.T + Minv[:, 2:]).T  # (N,2)
    return uv


# ------------------------- main criterion -------------------------

class TDOBBWKpt1Criterion(nn.Module):
    """
    Detection + single-keypoint criterion for YOLO11-style OBB + top-down keypoint.

    det_maps: list of 3 tensors [(B,C,H,W), ...] for strides (8,16,32)
              channel order (7 + nc): [tx,ty,tw,th,sin,cos,obj,(cls...)]
              IMPORTANT: we compute L1 on *log*-space tw/th (no exp in the loss).
    feats   : PAN/FPN features [P3, P4, P5] (P3 is used by ROI)
    batch   : supports either per-image lists {'bboxes','labels','kpts'} or a 'targets' tensor
              targets (M,8/9): [bix, cls, cx,cy,w,h,ang(rad), kpx,kpy]
    model   : must expose .kpt_from_obbs(feats, obb_list, scores_list=None)
              and its ROI meta provides 2x3 affine 'M' + 'feat_down'
    """

    def __init__(
        self,
        nc: Optional[int] = None,              # accept either 'nc' ...
        num_classes: Optional[int] = None,     # ... or 'num_classes' (builder handles both)  :contentReference[oaicite:2]{index=2}
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
    ) -> None:
        super().__init__()
        n_classes = num_classes if num_classes is not None else nc
        if n_classes is None:
            raise ValueError("Provide either 'nc' or 'num_classes' to TDOBBWKpt1Criterion")
        self.nc = int(n_classes)

        self.strides = tuple(int(s) for s in strides)
        assert len(self.strides) == 3, "expected 3 detection levels (P3/P4/P5)"

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
        self.level_boundaries = (float(level_boundaries[0]), float(level_boundaries[1]))

        # losses
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.smoothl1 = nn.SmoothL1Loss(reduction="mean")

    # --------------------- forward ---------------------

    def forward(self,
                det_maps: List[torch.Tensor],
                feats: Optional[List[torch.Tensor]],
                batch: Dict[str, Any],
                model: Optional[nn.Module] = None,
                epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, float]]:

        # -------- sanity checks --------
        assert isinstance(det_maps, (list, tuple)) and len(det_maps) == 3, \
            "det_maps must be a list of 3 tensors (P3,P4,P5)"
        B = det_maps[0].shape[0]
        # assert per-level channels = 7+nc
        if not hasattr(self, '_ASSERT_ONCE_DET'):
            self._ASSERT_ONCE_DET = False
        if not self._ASSERT_ONCE_DET:
            for i,dm in enumerate(det_maps):
                C = int(dm.shape[1]); expected = 7 + int(self.nc)
                assert C == expected, f"det_maps[{i}] has C={C}, expected {expected} (=7+nc)."
            self._ASSERT_ONCE_DET = True
        device = det_maps[0].device

        # channel split
        # [tx,ty,tw,th,sin,cos,obj,(cls...)]
        def split_maps(x: torch.Tensor) -> Dict[str, torch.Tensor]:
            c = x.shape[1]
            assert c >= 7, "expect at least 7 channels"
            tx, ty, tw, th, s, c_, obj = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6]
            cls = x[:, 7:] if c > 7 else None
            return {"tx": tx, "ty": ty, "tw": tw, "th": th, "sin": s, "cos": c_, "obj": obj, "cls": cls}

        P3, P4, P5 = det_maps
        m3, m4, m5 = split_maps(P3), split_maps(P4), split_maps(P5)
        if not hasattr(self, '_DBG_SINCOS_ONCE'):
            s3 = float(((m3['sin']**2 + m3['cos']**2).mean()).item())
            s4 = float(((m4['sin']**2 + m4['cos']**2).mean()).item())
            s5 = float(((m5['sin']**2 + m5['cos']**2).mean()).item())
            print(f"[ASSERT] sin^2+cos^2 mean: P3={s3:.3f} P4={s4:.3f} P5={s5:.3f}")
            self._DBG_SINCOS_ONCE = True

        # -------- parse GT (supports two formats) --------
        boxes_list, labels_list, kpts_list = self._read_targets(batch, B, device)

        
# -------- detection loss (obj, box, angle, cls) --------
# Angle sanity & auto-conversion: many datasets store angles in degrees by mistake.
if not hasattr(self, '_ANGLE_SANITY_ONCE'):
    self._ANGLE_SANITY_ONCE = True
    total, deg_like, wrapped = 0, 0, 0
    for bi, bl in enumerate(boxes_list):
        if bl is None or len(bl) == 0:
            continue
        ang = torch.as_tensor(bl, device=device, dtype=torch.float32)[:, 4]
        total += int(ang.numel())

        # (1) detect degree-like entries: |ang| in (pi, 360]
        mask_deg = (ang.abs() > 3.1415927 + 1e-3) & (ang.abs() <= 360.0 + 1e-3)
        if mask_deg.any():
            deg_like += int(mask_deg.sum().item())
            # convert ONLY those entries to radians in the original list
            bl = torch.as_tensor(bl, device=device, dtype=torch.float32)
            bl[mask_deg, 4] = bl[mask_deg, 4] * (3.1415927 / 180.0)
            boxes_list[bi] = bl

        # (2) wrap any remaining out-of-range radians into [-pi, pi]
        ang2 = torch.as_tensor(boxes_list[bi], device=device, dtype=torch.float32)[:, 4]
        mask_wrap = ang2.abs() > 3.1415927 + 1e-3
        if mask_wrap.any():
            wrapped += int(mask_wrap.sum().item())
            bl2 = torch.as_tensor(boxes_list[bi], device=device, dtype=torch.float32)
            # wrap: ((θ + π) mod 2π) − π
            bl2[mask_wrap, 4] = ((bl2[mask_wrap, 4] + 3.1415927) % (2.0 * 3.1415927)) - 3.1415927
            boxes_list[bi] = bl2

    if (deg_like > 0) or (wrapped > 0):
        print(f"[ANGLE-FIX] Converted {deg_like} GT angles from degrees and wrapped {wrapped} into [-pi,pi] out of {total} total.")
        l_obj, l_box, l_ang, l_cls, pos = self._loss_det((m3, m4, m5), (P3, P4, P5), boxes_list, labels_list)

        # -------- keypoint loss via ROI crops on P3 --------
        l_kpt, kpt_pos = self._loss_kpt(model, feats, boxes_list, kpts_list)

        # -------- total --------
        total = (self.lambda_obj * l_obj +
                 self.lambda_box * l_box +
                 self.lambda_ang * l_ang +
                 self.lambda_cls * l_cls +
                 self.lambda_kpt * l_kpt)

        logs = {
            "obj": float(_ddp_mean(l_obj.detach())),
            "box": float(_ddp_mean(l_box.detach())),
            "ang": float(_ddp_mean(l_ang.detach())),
            "cls": float(_ddp_mean(l_cls.detach())),
            "kpt": float(_ddp_mean(l_kpt.detach())),
            "pos": float(_ddp_mean(torch.tensor([pos], device=device))),
            "kpt_pos": float(_ddp_mean(torch.tensor([kpt_pos], device=device))),
            "total": float(_ddp_mean(total.detach())),
        }
        return total, logs

    # --------------------- helpers: targets ---------------------

    @torch.no_grad()
    def _read_targets(self, batch: Dict[str, Any], B: int, device: torch.device):
        """
        Returns 3 python lists (per-image):
            boxes_list[i]  -> Tensor (Ni,5) [cx,cy,w,h,ang(rad)]
            labels_list[i] -> LongTensor (Ni,)
            kpts_list[i]   -> Tensor (Ni,2) [x,y] in image px (optional; zero if missing)
        """
        if "targets" in batch and isinstance(batch["targets"], torch.Tensor):
            T = batch["targets"].to(device)
            if T.numel() == 0:
                return [torch.zeros(0, 5, device=device)] * B, \
                       [torch.zeros(0, dtype=torch.long, device=device)] * B, \
                       [torch.zeros(0, 2, device=device)] * B
            bix = T[:, 0].long()
            cls = T[:, 1].long() if T.shape[1] > 1 else torch.zeros_like(bix)
            cx, cy, w, h, ang = T[:, 2], T[:, 3], T[:, 4], T[:, 5], T[:, 6]
            has_kpt = T.shape[1] >= 9
            kx = T[:, 7] if has_kpt else torch.zeros_like(cx)
            ky = T[:, 8] if has_kpt else torch.zeros_like(cy)

            boxes_list = [torch.stack([cx[bix == i], cy[bix == i], w[bix == i], h[bix == i], ang[bix == i]], dim=1)
                          for i in range(B)]
            labels_list = [cls[bix == i] for i in range(B)]
            kpts_list = [torch.stack([kx[bix == i], ky[bix == i]], dim=1) if has_kpt else torch.zeros(0, 2, device=device)
                         for i in range(B)]
            return boxes_list, labels_list, kpts_list

        # per-image lists (preferred in your loader)
        boxes_list = [torch.as_tensor(b, device=device, dtype=torch.float32) for b in batch.get("bboxes", [])]
        labels_list = [torch.as_tensor(l, device=device, dtype=torch.long) for l in batch.get("labels", [])]
        kpts_raw = batch.get("kpts", [None] * len(boxes_list))

        # normalize shapes
        out_boxes, out_labels, out_kpts = [], [], []
        for i in range(B):
            bx = boxes_list[i] if i < len(boxes_list) else torch.zeros(0, 5, device=device)
            lb = labels_list[i] if i < len(labels_list) else torch.zeros(0, dtype=torch.long, device=device)
            kp = kpts_raw[i] if (kpts_raw is not None and i < len(kpts_raw) and kpts_raw[i] is not None) \
                 else torch.zeros(0, 2, device=device)
            bx = bx.reshape(-1, 5)
            lb = lb.reshape(-1)
            kp = torch.as_tensor(kp, device=device, dtype=torch.float32).reshape(-1, 2)
            out_boxes.append(bx)
            out_labels.append(lb)
            out_kpts.append(kp)
        return out_boxes, out_labels, out_kpts

    # --------------------- helpers: detection losses ---------------------

    def _route_level(self, max_side: float) -> int:
        low, mid = self.level_boundaries
        if max_side <= low:
            return 0
        elif max_side <= mid:
            return 1
        else:
            return 2

    def _loss_det(self,
                  maps_by_level: Tuple[Dict[str, torch.Tensor], ...],
                  raw_maps: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                  boxes_list: List[torch.Tensor],
                  labels_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:

        B = raw_maps[0].shape[0]
        device = raw_maps[0].device

        # running losses
        l_obj = raw_maps[0].new_zeros(())
        l_box = raw_maps[0].new_zeros(())
        l_ang = raw_maps[0].new_zeros(())
        l_cls = raw_maps[0].new_zeros(())
        total_pos = 0

        for b in range(B):
            boxes = boxes_list[b]  # (N,5)
            labels = labels_list[b] if self.nc > 1 else torch.zeros((boxes.shape[0],), dtype=torch.long, device=device)

            for n in range(boxes.shape[0]):
                cx, cy, w, h, ang = [float(x) for x in boxes[n]]
                max_side = max(w, h)
                lvl = self._route_level(max_side)
                stride = self.strides[lvl]
                mp = maps_by_level[lvl]  # dict of tensors with shape (B,H,W)

                # grid index
                H, W = mp["obj"].shape[-2], mp["obj"].shape[-1]
                gx, gy = cx / stride, cy / stride
                i, j = int(gx), int(gy)
                if not (0 <= i < W and 0 <= j < H):
                    continue  # object falls outside map

                # positive objectness target mask
                with torch.no_grad():
                    t_obj = torch.zeros_like(mp["obj"][b])
                    t_obj[j, i] = 1.0

                # objectness (mean over grid)
                l_obj = l_obj + self.bce(mp["obj"][b:b+1], t_obj.unsqueeze(0))

                # box regression (log-space tw/th, NO exp in loss)
                # predicted maps are logits; targets are log(w/stride), log(h/stride)
                tx_p = mp["tx"][b, j, i]
                ty_p = mp["ty"][b, j, i]
                tw_p = mp["tw"][b, j, i]
                th_p = mp["th"][b, j, i]

                tx_t = torch.tensor(gx - i, device=device, dtype=tx_p.dtype)
                ty_t = torch.tensor(gy - j, device=device, dtype=ty_p.dtype)
                tw_t = torch.tensor(math.log(max(w / stride, 1e-6)), device=device, dtype=tw_p.dtype)
                th_t = torch.tensor(math.log(max(h / stride, 1e-6)), device=device, dtype=th_p.dtype)

                l_box = l_box + self.smoothl1(tx_p, tx_t)
                l_box = l_box + self.smoothl1(ty_p, ty_t)
                l_box = l_box + self.smoothl1(tw_p, tw_t)
                l_box = l_box + self.smoothl1(th_p, th_t)

                # angle: normalize (sin,cos) before computing loss
                sin_p = mp["sin"][b, j, i]
                cos_p = mp["cos"][b, j, i]
                vec = torch.stack([sin_p, cos_p])
                vec = vec / (vec.norm(p=2) + 1e-6)
                sin_p_n, cos_p_n = vec[0], vec[1]
                sin_t = torch.tensor(math.sin(ang), device=device, dtype=sin_p.dtype)
                cos_t = torch.tensor(math.cos(ang), device=device, dtype=cos_p.dtype)
                l_ang = l_ang + self.smoothl1(sin_p_n, sin_t) + self.smoothl1(cos_p_n, cos_t)

                # class (optional)
                if maps_by_level[lvl]["cls"] is not None and self.nc > 1:
                    cls_logits = maps_by_level[lvl]["cls"][b, :, j, i]  # (nc,)
                    t = torch.full_like(cls_logits, 0.0)
                    t[int(labels[n].item())] = 1.0
                    l_cls = l_cls + F.binary_cross_entropy_with_logits(cls_logits, t)

                total_pos += 1

        # normalize by positives to keep scale stable
        norm = max(total_pos, 1)
        l_box = l_box / norm
        l_ang = l_ang / norm
        l_cls = l_cls / max(total_pos if self.nc > 1 else 1, 1)

        # l_obj already mean-reduced per level; average over batch images implicitly via loop
        l_obj = l_obj / max(total_pos, 1) if total_pos > 0 else l_obj

        return l_obj, l_box, l_ang, l_cls, total_pos

    # --------------------- helpers: keypoint loss ---------------------

    @torch.no_grad()
    def _roi_targets_from_meta(self,
                               metas: List[Dict[str, torch.Tensor]],
                               kpt_xy_img_list: List[torch.Tensor],
                               feat_down: float) -> Optional[torch.Tensor]:
        """
        Build uv targets (0..S in crop px) aligned with ROI metas for the corresponding GTs.
        metas[i]['M'] maps crop px -> feat px; we invert and map image kpt -> crop px.
        """
        if len(metas) == 0:
            return None
        uv_list = []
        for m in metas:
            assert 'M' in m, "ROI meta must include 2x3 affine 'M'"
            M = m['M']
            assert hasattr(M, 'shape') and tuple(M.shape[-2:])==(2,3), f"ROI meta 'M' must be (2,3), got {getattr(M, 'shape', None)}"
            if "gt_kpt" not in m or m["gt_kpt"] is None:
                # In training we attach per-ROI GT kpt below; if missing, skip
                return None
            kpt_xy = m["gt_kpt"]  # (1,2) tensor in image px
            M = m["M"]            # (2,3)
            uv = _img_kpts_to_crop_uv(kpt_xy, M, feat_down)  # (1,2) in crop px
            uv_list.append(uv)
        return torch.cat(uv_list, dim=0) if len(uv_list) else None

    def _loss_kpt(
            self,
            model,
            feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            boxes_list: Sequence[Union[torch.Tensor, np.ndarray, List[List[float]]]],
            kpts_list: Sequence[Union[torch.Tensor, np.ndarray, List[List[float]]]],
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute keypoint (u,v) loss from rotated ROI crops.

        - Supports metas from model.kpt_from_obbs either as:
            (a) per-batch list-of-lists of dicts, or
            (b) a flat list of dicts with 'bix' batch index
        - Expects exactly 1 keypoint per ROI (kpt1); if kpts have higher rank,
          the first keypoint is used.
        """
        import torch.nn.functional as F

        P3 = feats[0]
        device = P3.device
        dtype = P3.dtype
        B = int(P3.shape[0])

        # --- Normalize boxes_list to length B and to proper tensor shapes (N,5) ---
        norm_boxes_list: List[torch.Tensor] = []
        for b in range(B):
            if b < len(boxes_list) and boxes_list[b] is not None and len(boxes_list[b]) > 0:
                rois_b = torch.as_tensor(boxes_list[b], device=device, dtype=dtype)
                if rois_b.ndim == 1:
                    rois_b = rois_b.unsqueeze(0)  # (1,5)
                elif rois_b.ndim > 2:
                    rois_b = rois_b.reshape(-1, rois_b.shape[-1])
            else:
                rois_b = torch.zeros((0, 5), device=device, dtype=dtype)
            norm_boxes_list.append(rois_b)

        # Fast exit if really nothing to do
        if sum(int(x.shape[0]) for x in norm_boxes_list) == 0:
            zero = torch.zeros((), device=device, dtype=dtype)
            return zero, 0

        # --- Run model head to get predicted uv and metas describing each crop ---
        # NOTE: the model should be AMP-safe; uv_pred's dtype may be float16 under amp.
        uv_pred, metas = model.kpt_from_obbs(feats, norm_boxes_list, scores_list=None)

        # --- Normalize metas to per-batch lists and also build indices into uv_pred ---
        per_b_metas: List[List[dict]] = [[] for _ in range(B)]
        idxs_by_b: List[List[int]] = [[] for _ in range(B)]

        if isinstance(metas, (list, tuple)) and len(metas) > 0 and isinstance(metas[0], (list, tuple)):
            # metas is already per-batch lists
            start = 0
            for b in range(B):
                mb = list(metas[b]) if b < len(metas) else []
                per_b_metas[b] = mb
                idxs_by_b[b] = list(range(start, start + len(mb)))
                start += len(mb)
        else:
            # treat metas as a flat list[dict] with 'bix' keys
            flat = list(metas) if isinstance(metas, (list, tuple)) else []
            for i, md in enumerate(flat):
                bix = int(md.get("bix", md.get("batch_index", 0)))
                if 0 <= bix < B:
                    per_b_metas[bix].append(md)
                    idxs_by_b[bix].append(i)

        # --- Build loss ---
        total_loss = torch.zeros((), device=device, dtype=uv_pred.dtype)
        total_pos = 0

        for b in range(B):
            metas_b = per_b_metas[b]
            idxs_b = idxs_by_b[b]
            M_b = len(metas_b)
            if M_b == 0:
                continue

            # uv predictions for this batch image, keeping ROI order
            uv_b_pred = uv_pred[idxs_b]  # shape (M_b, 2)
            # Ground-truth keypoints for these ROIs
            # Accepts shapes: (M_b, 2) or (M_b, K, 2) or (M_b, >=2)
            kpts_b = torch.as_tensor(kpts_list[b], device=device, dtype=uv_b_pred.dtype)
            if kpts_b.ndim == 1:
                kpts_b = kpts_b.view(1, -1)
            # Reduce to (M_b, 2) using the first keypoint if needed
            if kpts_b.ndim == 3 and kpts_b.shape[1] >= 1:
                kpts_b = kpts_b[:, 0, :2]
            elif kpts_b.shape[-1] >= 2:
                kpts_b = kpts_b[:, :2]
            # Guard against count mismatch; trim/pad GT to match metas order
            if kpts_b.shape[0] != M_b:
                # Trim extra or pad missing with zeros (they will get low loss weight anyway)
                if kpts_b.shape[0] > M_b:
                    kpts_b = kpts_b[:M_b]
                else:
                    pad = torch.zeros((M_b - kpts_b.shape[0], 2), device=device, dtype=kpts_b.dtype)
                    kpts_b = torch.cat([kpts_b, pad], dim=0)

            # Compute GT (u,v) for each ROI using the affine meta
            uv_gt_list = []
            for j, md in enumerate(metas_b):
                M_crop_to_feat = torch.as_tensor(md["M"], device=device, dtype=uv_b_pred.dtype)  # (2,3)
                feat_down = float(md.get("feat_down", 8))
                xy_img = kpts_b[j:j + 1, :2]  # (1,2)
                uv_j = _img_kpts_to_crop_uv(xy_img, M_crop_to_feat, feat_down=feat_down)  # (1,2)
                uv_gt_list.append(uv_j)

            uv_b_gt = torch.cat(uv_gt_list, dim=0) if uv_gt_list else torch.zeros((0, 2), device=device,
                                                                                  dtype=uv_b_pred.dtype)

            # Smooth L1 over (u,v) in [0,1]; clamp GT softly just in case
            uv_b_gt = uv_b_gt.clamp(0.0, 1.0)
            # Align dtypes
            if uv_b_gt.dtype != uv_b_pred.dtype:
                uv_b_gt = uv_b_gt.to(uv_b_pred.dtype)

            loss_b = F.smooth_l1_loss(uv_b_pred, uv_b_gt, reduction="mean")
            total_loss = total_loss + loss_b
            total_pos += int(M_b)

        # Scale by configured lambda
        total_loss = total_loss * float(self.lambda_kpt)
        return total_loss, total_pos
