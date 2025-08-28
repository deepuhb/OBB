# src/models/losses/td_obb_kpt1_loss.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- small utilities -------------------------

def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()

def _ddp_mean(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().clone()
    if y.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        y = y.to(torch.float32)
    if _dist_ready():
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        y = y / float(dist.get_world_size())
    return y

def _ddp_sum(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().clone()
    if _dist_ready():
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y

def _safe_item(v: torch.Tensor, default: float = 0.0) -> float:
    try:
        return float(v.detach().item())
    except Exception:
        return float(default)

def _angle_wrap_le90(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-π/2, π/2)."""
    return torch.remainder(theta + math.pi / 2.0, math.pi) - math.pi / 2.0

def _nan_to_num_(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def _is_finite_tensor(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item() if isinstance(x, torch.Tensor) else True

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
    Convert GT keypoint (image px) -> crop uv (crop px).
    We: image px -> feature px (divide by feat_down) -> apply inverse(M).
    """
    if kpt_xy_img.dim() == 1:
        kpt_xy_img = kpt_xy_img.view(1, 2)
    xy_feat = kpt_xy_img / float(feat_down)
    Minv = _invert_affine_2x3(M_crop_to_feat)
    uv = (Minv[:, :2] @ xy_feat.T + Minv[:, 2:]).T  # (N,2)
    return uv


# ------------------------- DFL helpers -------------------------

def _dfl_target_bins(v: torch.Tensor, reg_max: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute adjacent-bin indices and linear weights for DFL.
    v: (N,) in [0, reg_max]
    Returns: li, ri, wl, wr
    """
    v_clamped = v.clamp(0, reg_max - 1e-6)
    li = torch.floor(v_clamped)
    ri = (li + 1).clamp(max=reg_max)
    wl = (ri - v_clamped)  # weight for left bin
    wr = (v_clamped - li)  # weight for right bin
    return li.long(), ri.long(), wl, wr

def _dfl_loss(logits: torch.Tensor, v: torch.Tensor, reg_max: int) -> torch.Tensor:
    """
    logits: (N, nbins) raw logits for a side (width OR height)
    v:      (N,) target value in [0, reg_max]
    """
    nbins = reg_max + 1
    assert logits.shape[1] == nbins
    li, ri, wl, wr = _dfl_target_bins(v, reg_max)
    logp = torch.log_softmax(logits, dim=1)   # (N, nbins)
    li_logp = logp.gather(1, li.unsqueeze(1)).squeeze(1)
    ri_logp = logp.gather(1, ri.unsqueeze(1)).squeeze(1)
    loss = -(wl * li_logp + wr * ri_logp)
    return loss.mean()

def _build_log_bins_from_head_or_default(model: Optional[nn.Module],
                                         strides: Sequence[int],
                                         device: torch.device,
                                         dtype: torch.dtype,
                                         reg_max: int) -> List[torch.Tensor]:
    """
    If the head exposes per-level (log_min, log_max), use those;
    otherwise fallback to a narrow, stride-aware range to avoid collapse.
    """
    bins: List[torch.Tensor] = []
    if hasattr(model, "head") and hasattr(model.head, "dfl_log_minmax"):
        # expected: list/tuple of (log_min, log_max) per level
        for li, s in enumerate(strides):
            vmin, vmax = model.head.dfl_log_minmax[li]
            bins.append(torch.linspace(float(vmin), float(vmax), steps=reg_max + 1,
                                       device=device, dtype=dtype))
        return bins

    # Fallback: widths/heights roughly ~ [0.5*s, 8*s] (per-level)
    for s in strides:
        log_min = math.log(max(0.5 * s, 1.0)) - math.log(s)  # log(w/stride)
        log_max = math.log(8.0 * s) - math.log(s)
        bins.append(torch.linspace(log_min, log_max, steps=reg_max + 1, device=device, dtype=dtype))
    return bins


# ------------------------- main criterion -------------------------

class TDOBBWKpt1Criterion(nn.Module):
    """
    **DFL-only** detection + (optional) single-kpt criterion for OBB.

    Maps per level (C = 2 + 2*(reg_max+1) + 1 + 1 + nc):
      [ tx, ty,
        dflw(0..R), dflh(0..R),
        ang_logit, obj, (cls...) ]

    - tx, ty: raw center offsets (we compare sigmoid(tx/ty) to the fractional cell offset target)
    - dflw/dflh: logits over 0..reg_max (targets built in log-space of (w/stride), same as decoder)
    - ang_logit: single logit; decoder maps to [-π/2, π/2); we use a cosine loss
    - obj, cls: BCE-with-logits (cls optional if nc=1)
    """

    def __init__(
        self,
        *,
        num_classes: Optional[int] = None,   # canonical
        nc: Optional[int] = None,            # accepted alias
        strides: Sequence[int] = (8, 16, 32),
        reg_max: int = 16,
        # loss weights
        lambda_box: float = 5.0,
        lambda_obj: float = 1.0,
        lambda_ang: float = 1.0,
        lambda_cls: float = 1.0,
        lambda_kpt: float = 0.0,             # off by default unless ROI/kpt path is wired
        # routing thresholds (in pixels for max(w,h))
        level_boundaries: Tuple[float, float] = (32.0, 64.0),  # <=32->P3, <=64->P4, else->P5
        # objectness negatives
        obj_neg_ratio: float = 3.0,          # neg:pos ratio (random subsampling)
        # angle range normalization (match decoder)
        ang_min: float = -math.pi / 2.0,
        ang_max: float =  math.pi / 2.0,
    ) -> None:
        super().__init__()
        n_classes = num_classes if num_classes is not None else nc
        if n_classes is None:
            raise ValueError("Provide 'num_classes' (or 'nc') to TDOBBWKpt1Criterion")

        self.nc = int(n_classes)
        self.strides = tuple(int(s) for s in strides)
        assert len(self.strides) == 3, "expected 3 detection levels (P3/P4/P5)"
        self.reg_max = int(reg_max)
        self.nbins = self.reg_max + 1

        # weights
        self.lambda_box = float(lambda_box)
        self.lambda_obj = float(lambda_obj)
        self.lambda_ang = float(lambda_ang)
        self.lambda_cls = float(lambda_cls)
        self.lambda_kpt = float(lambda_kpt)

        # misc
        self.level_boundaries = (float(level_boundaries[0]), float(level_boundaries[1]))
        self.obj_neg_ratio = float(obj_neg_ratio)
        self.ang_min = float(ang_min)
        self.ang_max = float(ang_max)

        # losses
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.smoothl1 = nn.SmoothL1Loss(reduction="mean")

        # one-time prints
        self._printed_ang_once = False
        self._assert_once = False

    # --------------------- public API ---------------------

    def forward(
        self,
        det_maps: List[torch.Tensor],
        feats: Optional[List[torch.Tensor]],
        batch: Dict[str, Any],
        model: Optional[nn.Module] = None,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        assert isinstance(det_maps, (list, tuple)) and len(det_maps) == 3, "need 3 detection levels"
        B = int(det_maps[0].shape[0])
        device = det_maps[0].device
        dtype = det_maps[0].dtype

        # Print angle-mode assertion once (rank-0)
        if not self._printed_ang_once:
            is_main = True
            try:
                if dist.is_available() and dist.isInitialized() and dist.get_rank() != 0:
                    is_main = False
            except Exception:
                pass
            if is_main:
                print("[ASSERT] angle uses single-logit (YOLO-style).")
            self._printed_ang_once = True

        # Consistency check (one-time)
        if not self._assert_once:
            expected = 2 + 2 * (self.reg_max + 1) + 1 + 1 + self.nc
            for i, m in enumerate(det_maps):
                C = int(m.shape[1])
                assert C == expected, (
                    f"det_maps[{i}] channels={C}, expected {expected}="
                    f"[tx,ty,dflw({self.nbins}),dflh({self.nbins}),ang,obj,(cls..)]"
                )
            self._assert_once = True

        # Parse targets
        gtb_list, gtc_list, gtk_list = self._read_targets(batch, B, device)

        # Build per-level log bins (from head if available, else fallback)
        log_bins = _build_log_bins_from_head_or_default(model, self.strides, device, dtype, self.reg_max)

        # Detection loss
        l_det, det_parts = self.det_loss(det_maps, gtb_list, gtc_list, log_bins)

        # Keypoint branch (optional, safe no-op if not wired)
        l_kpt, kpt_pos = self.kpt_loss(model, feats, gtb_list, gtk_list)

        total = l_det + l_kpt

        # Logs (DDP-mean)
        logs = {
            "obj": float(_ddp_mean(torch.tensor(det_parts.get("obj", 0.0), device=device))),
            "box": float(_ddp_mean(torch.tensor(det_parts.get("box", 0.0), device=device))),
            "dfl": float(_ddp_mean(torch.tensor(det_parts.get("dfl", 0.0), device=device))),
            "ang": float(_ddp_mean(torch.tensor(det_parts.get("ang", 0.0), device=device))),
            "cls": float(_ddp_mean(torch.tensor(det_parts.get("cls", 0.0), device=device))),
            "kpt": float(_ddp_mean(l_kpt.detach())),
            "pos": float(_ddp_sum(torch.tensor(det_parts.get("pos", 0), device=device))),
            "kpt_pos": float(_ddp_sum(torch.tensor(kpt_pos, device=device))),
            "total": float(_ddp_mean(total.detach())),
        }
        return total, logs

    # --------------------- parsing targets ---------------------

    @torch.no_grad()
    def _read_targets(self, batch: Dict[str, Any], B: int, device) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Returns:
          - gtb_list: per-image OBB (N_i,5) [cx,cy,w,h,ang]
          - gtc_list: per-image labels (N_i,)
          - gtk_list: per-image kpt (N_i,2) (optional, empty if absent)
        """
        gtb_list: List[torch.Tensor] = []
        gtc_list: List[torch.Tensor] = []
        gtk_list: List[torch.Tensor] = []

        if "targets" in batch and isinstance(batch["targets"], torch.Tensor):
            t = batch["targets"].to(device)
            # flexible columns: [bix, cls, cx,cy,w,h,ang,(kpx,kpy)?]
            cols = t.shape[1]
            assert cols >= 7, f"targets must have >=7 columns, got {cols}"
            for b in range(B):
                sel = (t[:, 0].long() == b)
                tb = t[sel]
                if tb.numel() == 0:
                    gtb_list.append(torch.zeros(0, 5, device=device, dtype=t.dtype))
                    gtc_list.append(torch.zeros(0, dtype=torch.long, device=device))
                    gtk_list.append(torch.zeros(0, 2, device=device, dtype=t.dtype))
                    continue
                cls = tb[:, 1].clamp_(min=0).long()
                cxcywhang = tb[:, 2:7]
                kpt = tb[:, 7:9] if cols >= 9 else torch.zeros(tb.shape[0], 2, device=device, dtype=t.dtype)
                gtb_list.append(cxcywhang)
                gtc_list.append(cls)
                gtk_list.append(kpt)
            return gtb_list, gtc_list, gtk_list

        # per-image lists (preferred)
        boxes = batch.get("bboxes", [])
        labels = batch.get("labels", [])
        kpts = batch.get("kpts", [])

        for b in range(B):
            bx = boxes[b].to(device) if len(boxes) > b and isinstance(boxes[b], torch.Tensor) else torch.zeros(0, 5, device=device)
            lb = labels[b].to(device).long() if len(labels) > b and isinstance(labels[b], torch.Tensor) else torch.zeros(bx.shape[0], device=device, dtype=torch.long)
            kp = kpts[b].to(device) if len(kpts) > b and isinstance(kpts[b], torch.Tensor) else torch.zeros(bx.shape[0], 2, device=device, dtype=bx.dtype)
            gtb_list.append(bx)
            gtc_list.append(lb)
            gtk_list.append(kp)
        return gtb_list, gtc_list, gtk_list

    # --------------------- map splitting ---------------------

    @staticmethod
    def _split_map(m: torch.Tensor, nbins: int, nc: int) -> Dict[str, torch.Tensor]:
        """
        Split (B,C,H,W) into:
          tx, ty, dflw(nbins), dflh(nbins), ang, obj, cls(nc?)
        """
        B, C, H, W = m.shape
        off = 0
        tx = m[:, off:off+1]; off += 1
        ty = m[:, off:off+1]; off += 1
        dflw = m[:, off:off+nbins]; off += nbins
        dflh = m[:, off:off+nbins]; off += nbins
        ang = m[:, off:off+1]; off += 1
        obj = m[:, off:off+1]; off += 1
        cls = m[:, off:off+nc] if nc > 1 else None
        return {"tx": tx, "ty": ty, "dflw": dflw, "dflh": dflh, "ang": ang, "obj": obj, "cls": cls}

    # --------------------- detection loss (DFL-only) ---------------------

    def det_loss(
        self,
        det_maps: List[torch.Tensor],
        gtb_list: List[torch.Tensor],
        gtc_list: List[torch.Tensor],
        log_bins: List[torch.Tensor],   # per-level linspace over log(w/stride), log(h/stride)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        device = det_maps[0].device
        dtype  = det_maps[0].dtype
        B = int(det_maps[0].shape[0])

        total_box = det_maps[0].new_tensor(0.0)
        total_dfl = det_maps[0].new_tensor(0.0)
        total_ang = det_maps[0].new_tensor(0.0)
        total_obj = det_maps[0].new_tensor(0.0)
        total_cls = det_maps[0].new_tensor(0.0)
        n_pos = 0

        for li, m in enumerate(det_maps):
            s = float(self.strides[li])
            H, W = int(m.shape[-2]), int(m.shape[-1])

            mp = self._split_map(m, self.nbins, self.nc)  # {'tx','ty','dflw','dflh','ang','obj','cls'}
            tx_map, ty_map = mp["tx"], mp["ty"]
            wlog_map, hlog_map = mp["dflw"], mp["dflh"]
            ang_map, obj_map = mp["ang"], mp["obj"]
            cls_map = mp["cls"]

            # Make grid indices for decoding fractional centers
            gy, gx = torch.meshgrid(
                torch.arange(H, device=device, dtype=dtype),
                torch.arange(W, device=device, dtype=dtype),
                indexing='ij'
            )
            gx = gx.view(1, 1, H, W)
            gy = gy.view(1, 1, H, W)

            # Build per-image positives for this level
            # Routing by size (in px): <=b0 -> P3; <=b1 -> P4; else -> P5
            b0, b1 = self.level_boundaries
            pos_idx: List[Tuple[int, int, int]] = []  # (b, j, i)
            targets_tx: List[float] = []
            targets_ty: List[float] = []
            targets_wv: List[float] = []  # DFL-scaled (0..reg_max) for width
            targets_hv: List[float] = []  # DFL-scaled (0..reg_max) for height
            targets_ang: List[float] = []
            targets_cls: List[int] = []

            # precompute bin scaling for this level
            bins = log_bins[li]                     # (nbins,)
            log_min = float(bins[0].item())
            log_max = float(bins[-1].item())
            # map log(w/stride) -> [0..reg_max]
            def _to_dfl_space(log_ws: torch.Tensor) -> torch.Tensor:
                t = (log_ws - log_min) / max(log_max - log_min, 1e-6) * float(self.reg_max)
                return t

            for b in range(B):
                gtb = gtb_list[b]  # (N,5) [cx,cy,w,h,ang] in image px
                gtc = gtc_list[b]  # (N,)
                if gtb.numel() == 0:
                    continue

                # size-based level routing
                maxsz = torch.max(gtb[:, 2], gtb[:, 3])  # (N,)
                if li == 0:
                    mask = (maxsz <= b0)
                elif li == 1:
                    mask = (maxsz > b0) & (maxsz <= b1)
                else:
                    mask = (maxsz > b1)

                if mask.sum().item() == 0:
                    continue

                sel = gtb[mask]
                cls = gtc[mask].clamp_(min=0, max=max(self.nc - 1, 0))
                cx, cy, gw, gh, ga = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3], sel[:, 4]

                # assign to cell
                gxv = cx / s
                gyv = cy / s
                ii = gxv.floor().long()
                jj = gyv.floor().long()

                # in-bounds only
                inb = (ii >= 0) & (jj >= 0) & (ii < W) & (jj < H)
                if inb.sum().item() == 0:
                    continue

                ii = ii[inb]; jj = jj[inb]
                ftx = gxv[inb] - ii.to(gxv.dtype)   # fractional [0,1)
                fty = gyv[inb] - jj.to(gyv.dtype)

                # DFL targets (log-space of (w/stride), (h/stride))
                logw = torch.log(gw[inb].clamp(min=1e-3) / s)
                logh = torch.log(gh[inb].clamp(min=1e-3) / s)
                wv = _to_dfl_space(logw)            # scaled [0..reg_max]
                hv = _to_dfl_space(logh)

                # angle (keep in [-π/2, π/2))
                ang_t = _angle_wrap_le90(ga[inb])

                # append
                for k in range(ii.numel()):
                    pos_idx.append((b, int(jj[k]), int(ii[k])))
                    targets_tx.append(float(ftx[k]))
                    targets_ty.append(float(fty[k]))
                    targets_wv.append(float(wv[k]))
                    targets_hv.append(float(hv[k]))
                    targets_ang.append(float(ang_t[k]))
                    targets_cls.append(int(cls[inb][k].item() if self.nc > 0 else 0))

            if len(pos_idx) == 0:
                # only negatives for obj
                # sample a small set of negatives to keep memory in check
                num_negs = max(int(0.01 * B * H * W), 1)
                b_rand = torch.randint(0, B, (num_negs,), device=device)
                j_rand = torch.randint(0, H, (num_negs,), device=device)
                i_rand = torch.randint(0, W, (num_negs,), device=device)
                obj_neg_logits = obj_map[b_rand, 0, j_rand, i_rand]
                total_obj = total_obj + self.bce(obj_neg_logits, torch.zeros_like(obj_neg_logits))
                continue

            n_lvl_pos = len(pos_idx); n_pos += n_lvl_pos

            # gather predictions at positives
            b_idx = torch.tensor([p[0] for p in pos_idx], device=device, dtype=torch.long)
            j_idx = torch.tensor([p[1] for p in pos_idx], device=device, dtype=torch.long)
            i_idx = torch.tensor([p[2] for p in pos_idx], device=device, dtype=torch.long)

            # centers (sigmoid to [0,1))
            tx_p = torch.sigmoid(tx_map[b_idx, 0, j_idx, i_idx])
            ty_p = torch.sigmoid(ty_map[b_idx, 0, j_idx, i_idx])

            tx_t = torch.tensor(targets_tx, device=device, dtype=dtype)
            ty_t = torch.tensor(targets_ty, device=device, dtype=dtype)

            total_box = total_box + self.smoothl1(tx_p, tx_t) + self.smoothl1(ty_p, ty_t)

            # DFL for width/height (logits -> (N, nbins))
            w_logits = wlog_map[b_idx, :, j_idx, i_idx].transpose(0, 1)  # (nbins, N) -> we want (N, nbins)
            h_logits = hlog_map[b_idx, :, j_idx, i_idx].transpose(0, 1)
            w_logits = w_logits.transpose(0, 1)  # (N, nbins)
            h_logits = h_logits.transpose(0, 1)

            wv_t = torch.tensor(targets_wv, device=device, dtype=dtype)
            hv_t = torch.tensor(targets_hv, device=device, dtype=dtype)

            total_dfl = total_dfl + _dfl_loss(w_logits, wv_t, self.reg_max)
            total_dfl = total_dfl + _dfl_loss(h_logits, hv_t, self.reg_max)

            # angle loss (map single logit -> angle via sigmoid to [0,1], then to [ang_min, ang_max])
            ang_logit = ang_map[b_idx, 0, j_idx, i_idx]
            ang_pred = torch.sigmoid(ang_logit) * (self.ang_max - self.ang_min) + self.ang_min
            ang_t = torch.tensor(targets_ang, device=device, dtype=dtype)
            total_ang = total_ang + (1.0 - torch.cos(ang_pred - ang_t)).mean()

            # objectness (pos=1)
            obj_pos_logits = obj_map[b_idx, 0, j_idx, i_idx]
            total_obj = total_obj + self.bce(obj_pos_logits, torch.ones_like(obj_pos_logits))

            # sample a few negatives for stabilization
            num_negs = int(self.obj_neg_ratio * n_lvl_pos)
            if num_negs > 0:
                b_neg = torch.randint(0, B, (num_negs,), device=device)
                j_neg = torch.randint(0, H, (num_negs,), device=device)
                i_neg = torch.randint(0, W, (num_negs,), device=device)
                obj_neg_logits = obj_map[b_neg, 0, j_neg, i_neg]
                total_obj = total_obj + self.bce(obj_neg_logits, torch.zeros_like(obj_neg_logits))

            # classification (pos only)
            if self.nc > 1 and cls_map is not None:
                cls_logits = cls_map[b_idx, :, j_idx, i_idx].transpose(0, 1)  # (N, nc)
                tcls = torch.tensor(targets_cls, device=device, dtype=torch.long)
                # one-hot BCEWithLogits (to match sigmoid decoder)
                t_onehot = F.one_hot(tcls, num_classes=self.nc).to(dtype)
                total_cls = total_cls + F.binary_cross_entropy_with_logits(cls_logits, t_onehot, reduction="mean")

        # normalize by positives (avoid div-by-zero)
        pos = max(n_pos, 1)
        loss = (
            self.lambda_box * (total_box / pos) +
            self.lambda_obj * (total_obj / pos) +
            self.lambda_ang * (total_ang / pos) +
            self.lambda_cls * (total_cls / pos) +
            # DFL already averaged; keep on the same scale as box
            (total_dfl / pos)
        )
        parts = {
            "box": _safe_item(total_box / pos),
            "dfl": _safe_item(total_dfl / pos),
            "ang": _safe_item(total_ang / pos),
            "obj": _safe_item(total_obj / pos),
            "cls": _safe_item(total_cls / pos),
            "pos": n_pos,
            "total": _safe_item(loss.detach()),
        }
        return loss, parts

    # --------------------- keypoint loss (optional) ---------------------

    def kpt_loss(
        self,
        model: Optional[nn.Module],
        feats: Optional[List[torch.Tensor]],
        gtb_list: List[torch.Tensor],
        gtk_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        """
        Very defensive: if ROI/kpt plumbing is not available in the model for this run,
        return a safe zero loss. If available, compute simple L1 on (u,v) in crop px.
        """
        if self.lambda_kpt <= 0.0:
            return torch.zeros((), device=gtb_list[0].device if len(gtb_list) else "cpu"), 0
        if model is None or feats is None or len(feats) == 0:
            return torch.zeros((), device=gtb_list[0].device if len(gtb_list) else "cpu"), 0

        device = feats[0].device
        l_kpt = torch.zeros((), device=device)
        kpt_pos = 0

        # Expect the forward code to have prepared per-ROI metas with affine 'M' and 'gt_kpt'
        metas: Optional[List[Dict[str, torch.Tensor]]] = getattr(model, "roi_metas", None)
        if not metas:
            return l_kpt, kpt_pos

        for m in metas:
            if "M" not in m or "gt_kpt" not in m:
                continue
            M = m["M"]
            gt_xy = m["gt_kpt"]
            feat_down = float(m.get("feat_down", 1.0))
            if not (hasattr(M, "shape") and tuple(M.shape[-2:]) == (2, 3)):
                continue
            if not (hasattr(gt_xy, "shape") and gt_xy.numel() == 2):
                continue

            uv_t = _img_kpts_to_crop_uv(gt_xy.reshape(1, 2).to(device=device, dtype=feats[0].dtype),
                                        M.to(device=device, dtype=feats[0].dtype),
                                        feat_down=feat_down)  # (1,2)
            # The model should provide predicted uv for this ROI (u,v) as a tensor
            if "pred_uv" not in m:
                # if not present, skip cleanly
                continue
            uv_p = m["pred_uv"].to(device=device, dtype=uv_t.dtype).reshape_as(uv_t)
            l_kpt = l_kpt + F.smooth_l1_loss(uv_p, uv_t, reduction="mean")
            kpt_pos += 1

        if kpt_pos == 0:
            return torch.zeros((), device=device), 0
        return self.lambda_kpt * (l_kpt / float(kpt_pos)), kpt_pos
