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

def _img_kpts_to_crop_uv(xy_img, M_crop_to_feat, feat_down=1):
    """
    Map 2D points to the feature (u,v) space using an affine that maps
    crop pixels -> feature coordinates. Supports M in shapes (2x2), (2x3), (3x3).
    xy_img can be shape (2,) or (N,2). Returns (N,2) in feature coords.

    NOTE: If your M already outputs coordinates in the *feature* grid,
    we still divide by feat_down only if feat_down != 1. If your M is in
    crop pixels and you want uv in the *downsampled* feature grid, keep feat_down as the layer's downsample (e.g. 8/16/32).
    """
    # ---- promote inputs to tensors and normalize shapes ----
    M = torch.as_tensor(M_crop_to_feat)  # do not detach; keep graph if needed
    if M.dim() != 2:
        raise ValueError(f"_img_kpts_to_crop_uv: expected 2D affine, got {M.shape}")
    dtype = M.dtype
    device = M.device

    xy = torch.as_tensor(xy_img, dtype=dtype, device=device)
    if xy.numel() == 2:
        xy = xy.reshape(1, 2)                  # (1,2)
    elif xy.dim() == 1 and xy.shape[0] != 2:
        raise ValueError(f"_img_kpts_to_crop_uv: 1D xy must be len=2, got {xy.shape}")
    elif xy.dim() == 2 and xy.shape[1] != 2:
        raise ValueError(f"_img_kpts_to_crop_uv: expected (N,2), got {xy.shape}")

    # ---- upgrade M to 3x3 homogeneous affine ----
    if M.shape == (2, 2):
        # [A|t] with t=0
        M23 = torch.cat([M, torch.zeros(2, 1, dtype=dtype, device=device)], dim=1)  # (2,3)
        M33 = torch.cat([M23, torch.tensor([[0., 0., 1.]], dtype=dtype, device=device)], dim=0)
    elif M.shape == (2, 3):
        M33 = torch.cat([M, torch.tensor([[0., 0., 1.]], dtype=dtype, device=device)], dim=0)
    elif M.shape == (3, 3):
        M33 = M
    else:
        raise ValueError(f"_img_kpts_to_crop_uv: unsupported M shape {tuple(M.shape)}; expected (2,2),(2,3),(3,3)")

    # ---- apply affine: target = [x y 1] @ M^T -> (N,2) ----
    ones = torch.ones((xy.shape[0], 1), dtype=dtype, device=device)
    xy1 = torch.cat([xy, ones], dim=1)       # (N,3)
    uv = (xy1 @ M33.T)[:, :2]               # (N,2)

    # ---- optional scale to feature grid if caller passes stride/downsample ----
    if isinstance(feat_down, (int, float)) and feat_down != 1:
        uv = uv / float(feat_down)

    # ---- strong sanity checks (trigger only on obvious bad cases) ----
    if torch.any(torch.isnan(uv)) or torch.any(torch.isinf(uv)):
        raise RuntimeError("[_img_kpts_to_crop_uv] produced non-finite uv; check your M and inputs.")
    return uv

def _wrap_pi(d: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]."""
    return (d + math.pi) % (2.0 * math.pi) - math.pi

def _le90_canonicalize(w: torch.Tensor, h: torch.Tensor, ang: torch.Tensor):
    """Ensure w >= h and rotate angle by +pi/2 where we swap; then wrap to [-pi/2, pi/2)."""
    swap = (w < h)
    w2 = torch.where(swap, h, w)
    h2 = torch.where(swap, w, h)
    ang2 = torch.where(swap, ang + math.pi / 2.0, ang)
    ang2 = torch.remainder(ang2 + math.pi / 2.0, math.pi) - math.pi / 2.0
    return w2, h2, ang2

def _clamp_hw_for_iou(w: torch.Tensor, h: torch.Tensor, img_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Prevent pathological IoU degenerate boxes."""
    w = w.clamp(min=1.0, max=img_max)
    h = h.clamp(min=1.0, max=img_max)
    return w, h

def _safe01(x: torch.Tensor) -> torch.Tensor:
    """Clamp to [0,1] and scrub NaN/Inf."""
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0, 1.0)

def _finite(x: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    """Replace NaN/Inf with a finite fill, keep dtype & device."""
    return torch.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)

def _angle_wrap_le90(theta: torch.Tensor) -> torch.Tensor:
    # wrap to [-pi/2, pi/2)
    return torch.remainder(theta + math.pi / 2.0, math.pi) - math.pi / 2.0

def _angle_periodic_loss(theta_p: torch.Tensor, theta_t: torch.Tensor) -> torch.Tensor:
    # 1 - cos(Δθ) is periodic and smooth
    return 1.0 - torch.cos(theta_p - theta_t)

def _dfl_soft_loss(logits: torch.Tensor, target: torch.Tensor, reg_max: int) -> torch.Tensor:
    """
    Distribution Focal Loss for a single scalar target.
    logits: (nbins,)
    target: scalar in [0, reg_max]; we use soft labels on floor/ceil.
    """
    nb = reg_max + 1
    t = target.clamp(0.0, float(reg_max) - 1e-6)
    l = torch.floor(t)
    r = l + 1.0
    wl = (r - t)
    wr = (t - l)
    l = int(l.item())
    r = int(min(float(r.item()), float(reg_max)))
    logp = torch.log_softmax(logits, dim=0)
    loss = - (wl * logp[l] + wr * logp[r])
    return loss


def _cells_in_radius(gx, gy, H, W, radius=2.5):
    # returns boolean mask (H, W) for cells within 'radius' cells of (gx, gy)
    ys = torch.arange(H, device=gx.device).float()
    xs = torch.arange(W, device=gx.device).float()
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    d2 = (xx - gx)**2 + (yy - gy)**2
    return d2 <= (radius * radius)

def _topk_indices_by_distance(gx, gy, mask, k=3):
    # from valid (mask==True) cells pick top-k closest to (gx,gy)
    if not mask.any():
        return None
    yy, xx = torch.where(mask)
    dx = (xx.float() - gx).abs()
    dy = (yy.float() - gy).abs()
    d = dx + dy
    k = min(k, d.numel())
    _, order = torch.topk(-d, k, largest=False)  # smallest distance
    return yy[order], xx[order]

def _safe_exp(x: torch.Tensor, max_log: float = 6.0) -> torch.Tensor:
    # cap input BEFORE exp to prevent inf (exp(6)=403, 7=1096)
    return torch.exp(x.clamp(min=-max_log, max=max_log))

def _is_finite_tensor(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()

def _nan_to_num_(x: torch.Tensor, val: float = 0.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=val, posinf=val, neginf=val)

def _dfl_target_bins(v: torch.Tensor, reg_max: int):
    """
    v: target scalar in [0, reg_max] (already divided by stride)
    returns (li, ri, wl, wr) with li<=v<=ri and wl+wr=1
    """
    v = v.clamp_(0, float(reg_max) - 1e-6)
    li = torch.floor(v)
    ri = li + 1.0
    wr = v - li
    wl = 1.0 - wr
    # clamp ri to reg_max and fix weights when li==reg_max
    ri = torch.min(ri, torch.tensor(float(reg_max), device=v.device, dtype=v.dtype))
    same = (ri == li)
    wl = torch.where(same, torch.ones_like(wl), wl)
    wr = torch.where(same, torch.zeros_like(wr), wr)
    return li.long(), ri.long(), wl, wr

def _dfl_loss(logits: torch.Tensor, v: torch.Tensor, reg_max: int) -> torch.Tensor:
    """
    logits: (N, nbins) raw logits for one side (here we use w or h)
    v:      (N,) target values in [0, reg_max]
    returns mean DFL loss
    """
    nbins = reg_max + 1
    assert logits.shape[1] == nbins
    li, ri, wl, wr = _dfl_target_bins(v, reg_max)
    # log-softmax for numerical stability
    logp = torch.log_softmax(logits, dim=1)  # (N, nbins)
    li_logp = logp.gather(1, li.unsqueeze(1)).squeeze(1)  # (N,)
    ri_logp = logp.gather(1, ri.unsqueeze(1)).squeeze(1)  # (N,)
    # linear interpolation between adjacent bins (DFL)
    loss = -(wl * li_logp + wr * ri_logp)
    return loss.mean()


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
        num_classes: Optional[int] = None,     # ... or 'num_classes' (builder handles both)
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
        """
        det_maps : list of 3 tensors [(B,C,H,W), ...] for strides (8,16,32)
        feats    : PAN/FPN features [P3,P4,P5]
        batch    : your loaded batch dict (supports both 'targets' tensor and per-image lists)
        model    : core model (needed for ROI kpt crops and DFL config if present)
        epoch    : training epoch (for schedules / debug)
        """
        # loss math in fp32 even if network ran in fp16
        if torch.is_autocast_enabled():
            det_maps = [dm.float() for dm in det_maps]

        assert isinstance(det_maps, (list, tuple)) and len(det_maps) == 3, \
            "det_maps must be a list of 3 tensors (P3,P4,P5)"
        device = det_maps[0].device
        B = int(det_maps[0].shape[0])

        # One-time sanity on channel layout & optional sin^2+cos^2 print
        if not hasattr(self, "_ASSERT_ONCE_DET"):
            self._ASSERT_ONCE_DET = False
        if not self._ASSERT_ONCE_DET:
            for i, dm in enumerate(det_maps):
                C = int(dm.shape[1])
                exp_classic = 7 + int(self.nc)
                exp_single = 6 + int(self.nc)
                ok = (C == exp_classic) or (C == exp_single) or (C >= 6)
                assert ok, f"det_maps[{i}] has C={C}, expected {exp_classic} (sin/cos) or {exp_single} (single ang) or DFL-compatible."

            # Print sin^2+cos^2 once (only if those channels exist)
            have_sincos = all(dm.shape[1] == (7 + self.nc) for dm in det_maps)
            if have_sincos and not hasattr(self, "_DBG_SINCOS_ONCE"):
                P3, P4, P5 = det_maps
                s3 = float(((P3[:, 4] ** 2 + P3[:, 5] ** 2).mean()).item())
                s4 = float(((P4[:, 4] ** 2 + P4[:, 5] ** 2).mean()).item())
                s5 = float(((P5[:, 4] ** 2 + P5[:, 5] ** 2).mean()).item())
                print(f"[ASSERT] sin^2+cos^2 mean: P3={s3:.3f} P4={s4:.3f} P5={s5:.3f}")
                self._DBG_SINCOS_ONCE = True
            self._ASSERT_ONCE_DET = True

        # Detection loss
        det_loss, det_parts = self._loss_det(det_maps, feats, batch, model=model, epoch=epoch)

        # Read targets and pass the CORRECT lists to kpt loss
        boxes_list, labels_list, kpts_list = self._read_targets(batch, B, device)

        # ROI keypoint loss (on P3 via model.kpt_from_obbs)
        l_kpt, kpt_pos = self._loss_kpt(model, feats, boxes_list, kpts_list)

        # Total
        total = det_loss + l_kpt  # _loss_kpt already includes self.lambda_kpt

        # Logs (DDP-reduced)
        logs = {
            "obj": float(_ddp_mean(torch.tensor(det_parts.get("obj", 0.0), device=device))),
            "box": float(_ddp_mean(torch.tensor(det_parts.get("box", 0.0), device=device))),
            "dfl": float(_ddp_mean(torch.tensor(det_parts.get("dfl", 0.0), device=device))),
            "ang": float(_ddp_mean(torch.tensor(det_parts.get("ang", 0.0), device=device))),
            "cls": float(_ddp_mean(torch.tensor(det_parts.get("cls", 0.0), device=device))),
            "kpt": float(_ddp_mean(l_kpt.detach())),
            "pos": float(_ddp_mean(torch.tensor([det_parts.get("pos", 0)], device=device))),
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

    @staticmethod
    def _ang_from_logit(z: torch.Tensor) -> torch.Tensor:
        # map logits -> [-pi/2, pi/2)
        return z.sigmoid() * math.pi - (math.pi / 2.0)

    def _split_map(self, dm: torch.Tensor, level: int, model: Optional[nn.Module]) -> Dict[str, torch.Tensor]:
        """
        Return a dict of per-channel maps for a given level:
        Classic: [tx,ty,tw,th,sin,cos,obj,(cls...)]
        Single-angle: [tx,ty,tw,th,ang,obj,(cls...)]
        DFL: [tx,ty, dflw(nb), dflh(nb), ang, obj, (cls...)] if model.reg_max is present and matches C.

        Shapes after this function:
          - tx, ty, (tw, th if present), ang, sin, cos : (B, H, W)  <-- always 3D for scalar channels
          - obj                                       : (B, 1, H, W)
          - cls                                       : (B, nc, H, W)  (if nc > 1)
          - dflw, dflh                                : (B, nbins, H, W)
        """
        C = int(dm.shape[1])
        nc = int(self.nc)
        mp: Dict[str, torch.Tensor] = {}

        # Try DFL first if model advertises reg_max
        reg_max = None
        if model is not None:
            reg_max = getattr(model, "reg_max", None)
            if reg_max is None and hasattr(model, "head"):
                reg_max = getattr(model.head, "reg_max", None)

        if isinstance(reg_max, int) and reg_max > 0:
            # expected layout: tx,ty | dflw(nb) | dflh(nb) | ang | obj | cls(nc)
            dflC = 2 + 2 * (reg_max + 1) + 1 + 1 + nc
            if C == dflC:
                i = 0
                mp["tx"] = dm[:, i, ...]
                i += 1  # (B,H,W)
                mp["ty"] = dm[:, i, ...]
                i += 1  # (B,H,W)
                mp["dflw"] = dm[:, i:i + (reg_max + 1), ...]
                i += (reg_max + 1)  # (B,nbins,H,W)
                mp["dflh"] = dm[:, i:i + (reg_max + 1), ...]
                i += (reg_max + 1)  # (B,nbins,H,W)
                mp["ang"] = dm[:, i, ...]
                i += 1  # (B,H,W)  <-- squeeze to 3D
                mp["obj"] = dm[:, i:i + 1, ...]
                i += 1  # (B,1,H,W)
                if nc > 1:
                    mp["cls"] = dm[:, i:i + nc, ...]  # (B,nc,H,W)
                return mp  # DFL path OK

        # Non-DFL paths
        if C == 7 + nc:
            # classic sin/cos
            i = 0
            mp["tx"] = dm[:, i, ...]
            i += 1  # (B,H,W)
            mp["ty"] = dm[:, i, ...]
            i += 1
            mp["tw"] = dm[:, i, ...]
            i += 1
            mp["th"] = dm[:, i, ...]
            i += 1
            mp["sin"] = dm[:, i, ...]
            i += 1
            mp["cos"] = dm[:, i, ...]
            i += 1
            mp["obj"] = dm[:, i:i + 1, ...]
            i += 1  # (B,1,H,W)
            if nc > 1:
                mp["cls"] = dm[:, i:i + nc, ...]
        else:
            # single-angle classic: [tx,ty,tw,th,ang,obj,(cls)]
            assert C >= 6, "invalid detection head channel layout"
            i = 0
            mp["tx"] = dm[:, i, ...]
            i += 1
            mp["ty"] = dm[:, i, ...]
            i += 1
            mp["tw"] = dm[:, i, ...]
            i += 1
            mp["th"] = dm[:, i, ...]
            i += 1
            mp["ang"] = dm[:, i, ...]
            i += 1  # (B,H,W)
            mp["obj"] = dm[:, i:i + 1, ...]
            i += 1  # (B,1,H,W)
            if nc > 1 and (i + nc) <= C:
                mp["cls"] = dm[:, i:i + nc, ...]
        return mp

    def _decode_at_cell(self, mp: Dict[str, torch.Tensor], b: int, j: int, i: int, stride: int,
                        dfl_bins: Optional[torch.Tensor] = None):
        # centers
        sx = mp["tx"][b, j, i].sigmoid()
        sy = mp["ty"][b, j, i].sigmoid()
        cx = (i + sx) * float(stride)
        cy = (j + sy) * float(stride)

        # sizes
        if dfl_bins is not None and "dflw" in mp and "dflh" in mp:
            pw_log = (mp["dflw"][b, :, j, i].softmax(dim=0) * dfl_bins).sum()
            ph_log = (mp["dflh"][b, :, j, i].softmax(dim=0) * dfl_bins).sum()
            pw = pw_log.exp() * float(stride)
            ph = ph_log.exp() * float(stride)
        else:
            pw = mp["tw"][b, j, i].exp().clamp_max(1e3) * float(stride)
            ph = mp["th"][b, j, i].exp().clamp_max(1e3) * float(stride)

        # angle
        if "ang" in mp:
            ang = self._ang_from_logit(mp["ang"][b, j, i])
        else:
            _s = mp["sin"][b, j, i]
            _c = mp["cos"][b, j, i]
            norm = (_s * _s + _c * _c).sqrt().clamp_min(1e-6)
            ang = torch.atan2(_s / norm, _c / norm)

        # LE-90 canonicalisation
        if pw < ph:
            pw, ph = ph, pw
            ang = ang + math.pi / 2.0
        ang = (ang + math.pi / 2.0) % math.pi - math.pi / 2.0
        return cx, cy, pw, ph, ang

    def _iou_rot(self, a, b):
        # a,b: [N,5] (cx,cy,w,h,theta radians)
        try:
            from mmcv.ops import box_iou_rotated
            A = a.clone()
            B = b.clone()
            A[:, 4] = torch.rad2deg(A[:, 4])
            B[:, 4] = torch.rad2deg(B[:, 4])
            return box_iou_rotated(A, B, aligned=False).clamp_(0, 1)
        except Exception:
            # AABB fallback
            def aabb(x):
                cx, cy, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
                return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

            A = a
            B = b
            a1 = aabb(A)
            b1 = aabb(B)
            out = torch.zeros((A.shape[0], B.shape[0]), device=A.device)
            for i in range(A.shape[0]):
                ax1, ay1, ax2, ay2 = a1[i]
                bx1, by1, bx2, by2 = b1.unbind(1)
                xx1 = torch.maximum(ax1, bx1)
                yy1 = torch.maximum(ay1, by1)
                xx2 = torch.minimum(ax2, bx2)
                yy2 = torch.minimum(ay2, by2)
                iw = (xx2 - xx1).clamp_min(0)
                ih = (yy2 - yy1).clamp_min(0)
                inter = iw * ih
                ua = (ax2 - ax1).clamp_min(0) * (ay2 - ay1).clamp_min(0)
                ub = (bx2 - bx1).clamp_min(0) * (by2 - by1).clamp_min(0)
                union = ua + ub - inter + 1e-9
                out[i] = inter / union
            return out.clamp_(0, 1)

    # --------------------- detection loss ---------------------
    def _loss_det(self, det_maps, feats, batch, model=None, epoch=None):
        """
        YOLO11-style OBB loss.

        Supports:
          - classic: [tx,ty,tw,th,ang,obj,(cls...)]
          - DFL:     [tx,ty,dflw(nb),dflh(nb),ang,obj,(cls...)]

        Box regression:
          * classic -> SmoothL1 on log(w/stride), log(h/stride)
          * DFL     -> (a) SmoothL1 on expected log-size (stabilizer)
                       (b) DFL soft-label CE on the per-side distributions

        Angle: periodic 1 - cos(Δθ). Objectness: target = IoU (warmup to 0.5+0.5*IoU early).
        """
        import math
        import torch
        import torch.nn.functional as F

        device = det_maps[0].device
        B = det_maps[0].shape[0]

        # Split levels into per-channel dicts; preserves your key names
        mp_levels = [self._split_map(dm, li, model=model) for li, dm in enumerate(det_maps)]
        Hs = [int(dm.shape[-2]) for dm in det_maps]
        Ws = [int(dm.shape[-1]) for dm in det_maps]
        strides = tuple(int(s) for s in self.strides)

        # Read GT (per-image lists)
        gtb_list, gtc_list, _ = self._read_targets(batch, B, device)

        # Image size (for sane caps)
        if isinstance(batch, dict) and "imgs" in batch and hasattr(batch["imgs"], "shape"):
            Himg = int(batch["imgs"].shape[-2])
            Wimg = int(batch["imgs"].shape[-1])
        else:
            Himg, Wimg = 640, 640
        img_max = float(max(Himg, Wimg))

        # Level routing thresholds (<=low -> P3, <=mid -> P4, else -> P5)
        low, mid = self.level_boundaries
        lvl_max_pix = [float(low), float(mid), img_max]

        # Detect DFL config
        reg_max = None
        if model is not None:
            reg_max = getattr(model, "reg_max", None)
            if reg_max is None and hasattr(model, "head"):
                reg_max = getattr(model.head, "reg_max", None)
        use_dfl = isinstance(reg_max, int) and reg_max > 0
        nb = (reg_max + 1) if use_dfl else None

        # Per-level log-bins for DFL (cover [log(1/stride), log(level_cap/stride)])
        log_bins_per_level = [None, None, None]
        if use_dfl:
            for li, s in enumerate(strides):
                log_min = math.log(1.0 / float(s))
                log_max = math.log(max(2.0, lvl_max_pix[li]) / float(s))
                idx = torch.arange(nb, device=device, dtype=torch.float32)
                log_bins_per_level[li] = log_min + (log_max - log_min) * (idx / float(reg_max))

        # Helpers
        def _nan2(x, v=0.0):
            return torch.nan_to_num(x, nan=v, posinf=v, neginf=v)

        def _ang_from_logit(z):  # [-pi/2, pi/2)
            return (torch.sigmoid(z) * math.pi) - (math.pi / 2.0)

        def _le90(w, h, a):
            swap = (w < h)
            w2 = torch.where(swap, h, w)
            h2 = torch.where(swap, w, h)
            a2 = a + swap.to(a.dtype) * (math.pi / 2.0)
            a2 = (a2 + math.pi / 2.0) % math.pi - math.pi / 2.0
            return w2, h2, a2

        def _aabb_iou(cx, cy, w, h, Cx, Cy, W, H):
            x1 = cx - 0.5 * w;
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w;
            y2 = cy + 0.5 * h
            X1 = Cx - 0.5 * W;
            Y1 = Cy - 0.5 * H
            X2 = Cx + 0.5 * W;
            Y2 = Cy + 0.5 * H
            xx1 = torch.maximum(x1, X1);
            yy1 = torch.maximum(y1, Y1)
            xx2 = torch.minimum(x2, X2);
            yy2 = torch.minimum(y2, Y2)
            iw = (xx2 - xx1).clamp_min(0);
            ih = (yy2 - yy1).clamp_min(0)
            inter = iw * ih
            union = (w.clamp_min(0) * h.clamp_min(0) + W.clamp_min(0) * H.clamp_min(0) - inter + 1e-9)
            return (inter / union).clamp_(0, 1)

        def _angle_loss(ang_p, ang_t):
            d = (ang_p - ang_t + math.pi) % (2 * math.pi) - math.pi
            return 1.0 - torch.cos(d)

        def _closest_cells(gx, gy, H, W, radius=2.5, K=3):
            r = int(max(1, math.floor(radius)))
            x0 = max(0, int(math.floor(float(gx))) - r)
            x1 = min(W - 1, int(math.floor(float(gx))) + r)
            y0 = max(0, int(math.floor(float(gy))) - r)
            y1 = min(H - 1, int(math.floor(float(gy))) + r)
            if x1 < x0 or y1 < y0: return []
            xs = torch.arange(x0, x1 + 1, device=device)
            ys = torch.arange(y0, y1 + 1, device=device)
            xx, yy = torch.meshgrid(xs, ys, indexing='xy')
            d = (xx.float() - gx).abs() + (yy.float() - gy).abs()
            d = d.reshape(-1)
            yyf = yy.reshape(-1).long()
            xxf = xx.reshape(-1).long()
            k = min(K, d.numel())
            if k == 0: return []
            _, order = torch.topk(-d, k, largest=False)
            return [(int(yyf[o].item()), int(xxf[o].item())) for o in order]

        # Warmup for objectness target
        warm_epochs = 10
        use_warm = (epoch is not None) and (epoch <= warm_epochs)

        # Accumulators
        l_obj = det_maps[0].new_zeros(())
        l_box = det_maps[0].new_zeros(())
        l_ang = det_maps[0].new_zeros(())
        l_cls = det_maps[0].new_zeros(())
        l_dfl = det_maps[0].new_zeros(())
        pos_cnt = 0

        for b in range(B):
            gtb = gtb_list[b] if b < len(gtb_list) else torch.zeros(0, 5, device=device)
            gtc = gtc_list[b] if b < len(gtc_list) else torch.zeros(0, dtype=torch.long, device=device)

            if gtb.numel() == 0:
                # Negatives: random sample a few positions per level
                for li in range(len(mp_levels)):
                    obj_map = mp_levels[li]["obj"][b, 0]  # (H,W)
                    H, W = obj_map.shape
                    if H * W == 0: continue
                    n_neg = min(256, H * W)
                    idx = torch.randint(0, H * W, (n_neg,), device=device)
                    neg = _nan2(obj_map.view(-1)[idx])
                    l_obj = l_obj + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg), reduction="mean")
                continue

            if self.nc > 0:
                gtc = gtc.clamp_(0, max(0, self.nc - 1))

            for k in range(int(gtb.shape[0])):
                cx, cy, w, h, ang = gtb[k]
                w, h, ang = _le90(w, h, ang)

                # Route to level by object size
                max_side = float(torch.maximum(w, h))
                li = 0 if max_side <= low else (1 if max_side <= mid else 2)
                stride = strides[li]
                mp = mp_levels[li]
                H, W = Hs[li], Ws[li]

                # Feature-space center
                gx = cx / float(stride)
                gy = cy / float(stride)
                cells = _closest_cells(gx, gy, H, W, radius=2.5, K=3)
                if not cells:
                    continue

                # Targets in map space (log-sizes and angle)
                tw_t = torch.log((w / float(stride)).clamp(min=1e-6))
                th_t = torch.log((h / float(stride)).clamp(min=1e-6))
                ang_t = ang

                for (jj, ii) in cells:
                    # Center offsets (pred)
                    tx_p = torch.sigmoid(_nan2(mp["tx"][b, jj, ii]))
                    ty_p = torch.sigmoid(_nan2(mp["ty"][b, jj, ii]))

                    # Sizes
                    if use_dfl and ("dflw" in mp) and ("dflh" in mp):
                        # expected log-width/height from distributions
                        log_bins = log_bins_per_level[li]  # (nb,)
                        logits_w = _nan2(mp["dflw"][b, :, jj, ii])
                        logits_h = _nan2(mp["dflh"][b, :, jj, ii])
                        probs_w = torch.log_softmax(logits_w, dim=0).exp()
                        probs_h = torch.log_softmax(logits_h, dim=0).exp()
                        tw_exp = (probs_w * log_bins).sum()
                        th_exp = (probs_h * log_bins).sum()

                        # (a) small stabilizer on expected value
                        l_box = l_box + F.smooth_l1_loss(tw_exp, tw_t, reduction="mean")
                        l_box = l_box + F.smooth_l1_loss(th_exp, th_t, reduction="mean")

                        # (b) DFL soft-label CE (create two-bin targets right here)
                        log_min = float(log_bins[0].item())
                        log_max = float(log_bins[-1].item())
                        scale = reg_max / max(1e-6, (log_max - log_min))
                        tbin_w = (float(tw_t.item()) - log_min) * scale
                        tbin_h = (float(th_t.item()) - log_min) * scale
                        l_dfl = l_dfl + _dfl_soft_loss(logits_w, torch.tensor(tbin_w, device=device), reg_max)
                        l_dfl = l_dfl + _dfl_soft_loss(logits_h, torch.tensor(tbin_h, device=device), reg_max)

                        # Decode sizes for IoU target
                        w_p = torch.exp(tw_exp).clamp(min=1.0, max=1e4) * float(stride)
                        h_p = torch.exp(th_exp).clamp(min=1.0, max=1e4) * float(stride)
                    else:
                        # classic tw/th path
                        tw_p = _nan2(mp["tw"][b, jj, ii])
                        th_p = _nan2(mp["th"][b, jj, ii])

                        # SmoothL1 on log-sizes
                        l_box = l_box + F.smooth_l1_loss(tw_p, tw_t, reduction="mean")
                        l_box = l_box + F.smooth_l1_loss(th_p, th_t, reduction="mean")

                        # Decode for IoU target
                        w_p = torch.exp(tw_p).clamp(min=1.0, max=1e4) * float(stride)
                        h_p = torch.exp(th_p).clamp(min=1.0, max=1e4) * float(stride)

                    # Angle loss
                    if "ang" in mp:
                        ang_p = _ang_from_logit(_nan2(mp["ang"][b, jj, ii]))
                    elif ("sin" in mp) and ("cos" in mp):
                        s = _nan2(mp["sin"][b, jj, ii]);
                        c = _nan2(mp["cos"][b, jj, ii])
                        norm = torch.sqrt(torch.clamp(s * s + c * c, min=1e-12))
                        ang_p = torch.atan2(s / norm, c / norm)
                    else:
                        ang_p = ang_t
                    # Canonicalize pred box (LE-90)
                    if w_p < h_p:
                        w_p, h_p = h_p, w_p
                        ang_p = ang_p + math.pi / 2.0
                    ang_p = (ang_p + math.pi / 2.0) % math.pi - math.pi / 2.0
                    l_ang = l_ang + _angle_loss(ang_p, ang_t)

                    # Classification (only if multi-class)
                    if self.nc > 1 and "cls" in mp:
                        cvec = _nan2(mp["cls"][b, :, jj, ii])
                        tgt = torch.zeros_like(cvec);
                        tgt[int(gtc[k].item())] = 1.0
                        l_cls = l_cls + F.binary_cross_entropy_with_logits(cvec, tgt)

                    # Objectness with IoU target (warmup)
                    cx_p = (ii + tx_p) * float(stride)
                    cy_p = (jj + ty_p) * float(stride)
                    with torch.no_grad():
                        iou = _aabb_iou(cx_p, cy_p, w_p, h_p, cx, cy, w, h)
                        obj_t = (0.5 + 0.5 * iou) if use_warm else iou
                        obj_t = obj_t.clamp(0.0, 1.0)
                    obj_logit = _nan2(mp["obj"][b, 0, jj, ii])
                    l_obj = l_obj + F.binary_cross_entropy_with_logits(obj_logit, obj_t)

                    pos_cnt += 1

                # Light “ring” negatives around matched cells
                obj_map = mp["obj"][b, 0]
                gi, gj = int(math.floor(float(gx))), int(math.floor(float(gy)))
                rr = 3;
                ring = []
                for yy in range(max(0, gj - rr), min(H - 1, gj + rr) + 1):
                    for xx in range(max(0, gi - rr), min(W - 1, gi + rr) + 1):
                        # avoid exact matched cells
                        if (yy, xx) not in cells:
                            ring.append((yy, xx))
                if ring:
                    pick = min(8, len(ring))
                    sel = [ring[int(i.item())] for i in torch.randint(0, len(ring), (pick,), device=device)]
                    obj_neg = torch.stack([_nan2(obj_map[y, x]) for (y, x) in sel])
                    l_obj = l_obj + F.binary_cross_entropy_with_logits(obj_neg, torch.zeros_like(obj_neg),
                                                                       reduction="mean")

        # Normalize by positives
        pos = max(1, pos_cnt)
        l_obj = l_obj / pos
        l_box = l_box / pos
        l_ang = l_ang / pos
        l_cls = l_cls / pos
        l_dfl = l_dfl / pos

        total = (self.lambda_obj * l_obj
                 + self.lambda_box * l_box
                 + self.lambda_ang * l_ang
                 + self.lambda_cls * l_cls
                 + l_dfl)  # keep DFL weight at 1.0

        parts = {
            "obj": float(l_obj.detach()),
            "box": float(l_box.detach()),
            "ang": float(l_ang.detach()),
            "cls": float(l_cls.detach()),
            "dfl": float(l_dfl.detach()),
            "pos": float(pos_cnt),
            "kpt": 0.0,
            "kpt_pos": 0.0,
            "total": float(total.detach()),
        }
        return total, parts

    # --------------------- helpers: keypoint loss ---------------------
    @torch.no_grad()
    def _roi_targets_from_meta(self,
                               metas: List[Dict[str, torch.Tensor]],
                               kpt_xy_img_list: List[torch.Tensor],
                               feat_down: float) -> Optional[torch.Tensor]:
        """
        Build UV targets in feature-grid coordinates aligned with ROI metas.

        Each meta dict must include:
          - 'M': (2,3) affine mapping crop pixels -> feature pixels
        We take the *corresponding* keypoint from kpt_xy_img_list[j] for meta j.
        """
        if not metas:
            return None

        uv_list = []
        for j, m in enumerate(metas):
            if 'M' not in m:
                raise KeyError("ROI meta must include affine 'M' (2x3) mapping crop->feat.")
            M = m['M']
            if not (hasattr(M, "shape") and tuple(M.shape[-2:]) == (2, 3)):
                raise AssertionError(f"ROI meta 'M' must be (2,3), got {getattr(M, 'shape', None)}")

            # pick j-th keypoint (image px)
            if j >= len(kpt_xy_img_list):
                # pad missing with zeros to avoid crashing
                kpt_xy = torch.zeros(1, 2, device=M.device, dtype=M.dtype)
            else:
                kpt_xy = torch.as_tensor(kpt_xy_img_list[j], device=M.device, dtype=M.dtype).reshape(1, 2)

            # M already maps to feature pixels -> don't divide again
            uv = _img_kpts_to_crop_uv(kpt_xy, M, feat_down=1)
            uv_list.append(uv)

        return torch.cat(uv_list, dim=0) if uv_list else None

    def _loss_kpt(
            self,
            model: nn.Module,
            feats: Optional[List[torch.Tensor]],
            gtb_list: List[torch.Tensor],
            gtc_list: List[torch.Tensor],
    ):
        # If kpt head isn’t used or feats unavailable, be a no-op
        if not hasattr(model, "kpt_enabled") and feats is None:
            return feats[0].new_zeros(()), 0  # safe default
        if feats is None or len(feats) == 0:
            return torch.zeros((), device=gtb_list[0].device if len(gtb_list) else "cpu"), 0

        device = feats[0].device
        l_kpt = torch.zeros((), device=device)
        kpt_pos = 0

        # Your code attaches ROI metas in forward; pull them defensively if present
        metas: List[Dict[str, torch.Tensor]] = getattr(model, "roi_metas", None)
        if metas is None or len(metas) == 0:
            return l_kpt, kpt_pos  # nothing to train this step

        # Expected interface (kept from your file): each meta has keys
        # 'M' (2x3 affine crop->feat) and 'gt_kpt' (1,2) in image px.
        valid = 0
        for m in metas:
            if "M" not in m or "gt_kpt" not in m:
                continue
            M = m["M"]
            kpt_xy = m["gt_kpt"]
            if not (hasattr(M, "shape") and tuple(M.shape[-2:]) == (2, 3)):
                continue
            if not (hasattr(kpt_xy, "shape") and kpt_xy.numel() == 2):
                continue

            # (1,2)
            xy = kpt_xy.reshape(1, 2).to(device, dtype=feats[0].dtype)
            M = M.to(device, dtype=feats[0].dtype)

            # Map image->crop px using the inverse of (crop->feat) then scale by feat_down
            # But your helper already expects M that maps crop->feat; we only need crop u,v targets.
            # Use your existing helper if present; else do a minimal safe transform:
            try:
                uv = _img_kpts_to_crop_uv(xy, M, feat_down=getattr(self, "feat_down", 1.0))  # (1,2)
            except Exception:
                # very defensive: treat as identity if it fails
                uv = xy.clone()

            uv = _nan_to_num_(uv)
            if not _is_finite_tensor(uv):
                continue

            # Predicted uv logit maps should be in model (e.g., model.kpt_head output).
            # If not available, skip safely.
            if not hasattr(model, "kpt_predictor") or model.kpt_predictor is None:
                continue

            pred_u, pred_v = model.kpt_predictor(  # expected to return logits or coords
                feats
            )
            # Accept either logits (same shape) or scalar per-ROI predictions
            if isinstance(pred_u, torch.Tensor) and isinstance(pred_v, torch.Tensor):
                pu = _nan_to_num_(pred_u.squeeze())
                pv = _nan_to_num_(pred_v.squeeze())
                # match shapes
                if pu.dim() == 0 and pv.dim() == 0:
                    pu = pu.view(1)
                    pv = pv.view(1)
                if pu.numel() != 1 or pv.numel() != 1:
                    # cannot align confidently; skip this ROI
                    continue
                # simple L1 in crop space
                l_kpt = l_kpt + (pu - uv[:, 0]).abs().mean() + (pv - uv[:, 1]).abs().mean()
                kpt_pos += 1
            else:
                # predictor not ready; skip
                continue

            valid += 1

        if kpt_pos > 0:
            l_kpt = l_kpt / float(kpt_pos)

        return l_kpt, kpt_pos

