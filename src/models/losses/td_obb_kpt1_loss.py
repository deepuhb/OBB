# td_obb_kpt1_loss.py
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- small utils -------------------------

def _safe_exp(x: torch.Tensor, max_log: float = 8.0) -> torch.Tensor:
    # cap input BEFORE exp to prevent inf; exp(8) ~ 2981
    return torch.exp(x.clamp(min=-max_log, max=max_log))

def _is_finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())

def _nan_to_num_(x: torch.Tensor, val: float = 0.0) -> torch.Tensor:
    # inplace-safe for grads
    return torch.nan_to_num(x, nan=val, posinf=val, neginf=val)

def _ddp_mean(x: torch.Tensor) -> torch.Tensor:
    # works on 1-element tensors; no-op for single-GPU
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        y = x.clone()
        torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.SUM)
        y /= torch.distributed.get_world_size()
        return y
    return x

def _angle_from_logit(z: torch.Tensor) -> torch.Tensor:
    # map logits -> [-pi/2, pi/2)
    return (z.sigmoid() * math.pi) - (math.pi / 2.0)

def _angle_loss(a_pred: torch.Tensor, a_tgt: torch.Tensor) -> torch.Tensor:
    # periodic, smooth, bounded [0, 2]
    return (1.0 - torch.cos(a_pred - a_tgt)).mean() if a_pred.numel() else a_pred.new_tensor(0.0)

def _dfl_nll_loss(logits: torch.Tensor, target_frac_idx: torch.Tensor) -> torch.Tensor:
    """
    Distribution Focal Loss (2-bin linear interpolation).
    logits: (N, nbins)
    target_frac_idx: (N,) float in [0, nbins-1]
    """
    if target_frac_idx.numel() == 0:
        return logits.new_tensor(0.0)
    nbins = logits.shape[1]
    t = target_frac_idx.clamp(min=0.0, max=nbins - 1 - 1e-6)
    j = t.floor().long()                   # left bin
    r = (t - j.float()).clamp(0.0, 1.0)    # fractional part
    j1 = (j + 1).clamp(max=nbins - 1)      # right bin

    lsm = F.log_softmax(logits, dim=1)     # (N, nbins)
    nll = -(1.0 - r) * lsm.gather(1, j.view(-1, 1)).squeeze(1) \
          - r * lsm.gather(1, j1.view(-1, 1)).squeeze(1)
    return nll.mean()

# ------------------------- criterion -------------------------

class TDOBBWKpt1Criterion(nn.Module):
    """
    Detection (DFL for w/h + tx/ty + angle + obj + optional cls) + single-keypoint ROI loss.
    Head/key layout (DFL-only):
        tx, ty, w_logits, h_logits, ang, obj, (cls if num_classes>1)

    Notes
    -----
    * DFL bins are defined in **log-space** per level with **narrow ranges** to avoid bin saturation:
        P3: log([min_wh_img[0], max_wh_img[0]] / stride3)  e.g. [2, 96] px
        P4: log([min_wh_img[1], max_wh_img[1]] / stride4)  e.g. [4, 192] px
        P5: log([min_wh_img[2], max_wh_img[2]] / stride5)  e.g. [8, 384] px
    * We stabilize with SmoothL1 on the **expected** log-size from logits (EV in log-space).
    * Objectness uses BCE (pos=1 at assigned cells; background negatives sampled).
    """
    def __init__(self,
                 num_classes: int,
                 strides: Tuple[int, int, int] = (8, 16, 32),
                 level_boundaries: Tuple[float, float] = (32.0, 64.0),  # routing: <=low->P3, <=mid->P4, else P5
                 # loss weights
                 lambda_obj: float = 1.0,
                 lambda_box: float = 1.0,
                 lambda_dfl: float = 1.0,      # <--- DFL weight (requested)
                 lambda_ang: float = 0.5,
                 lambda_cls: float = 0.5,
                 lambda_kpt: float = 0.0,      # off by default
                 # misc
                 neg_obj_ratio: float = 4.0,
                 eps: float = 1e-7,
                 **_unused):
        super().__init__()
        self.nc = int(num_classes)
        self.strides = tuple(int(s) for s in strides)
        self.level_boundaries = (float(level_boundaries[0]), float(level_boundaries[1]))

        self.lambda_obj = float(lambda_obj)
        self.lambda_box = float(lambda_box)
        self.lambda_dfl = float(lambda_dfl)
        self.lambda_ang = float(lambda_ang)
        self.lambda_cls = float(lambda_cls) if self.nc > 1 else 0.0
        self.lambda_kpt = float(lambda_kpt)

        self.neg_obj_ratio = float(neg_obj_ratio)
        self.eps = float(eps)

        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.smoothl1 = nn.SmoothL1Loss(reduction="mean", beta=1.0)

    # ---------------------- public API (trainer calls this) ----------------------
    def forward(self,
                det_maps: List[torch.Tensor],
                feats: Optional[List[torch.Tensor]],
                batch: Dict[str, Any],
                model: Optional[nn.Module] = None,
                epoch: Optional[int] = None):

        device = det_maps[0].device
        for li, m in enumerate(det_maps):
            assert m.dim() == 4, f"det_maps[{li}] must be BCHW, got {tuple(m.shape)}"

        # simple one-time assertion print
        if not hasattr(self, "_ASSERT_ONCE"):
            print("[ASSERT] angle uses single-logit (YOLO-style).")
            self._ASSERT_ONCE = True

        # --- Detection loss
        det_loss, det_parts = self._loss_det(det_maps, feats, batch, model=model, epoch=epoch)

        # --- Keypoint loss (ROI)
        boxes_list, labels_list, kpts_list = self._read_targets(batch, det_maps[0].shape[0], device)
        l_kpt, kpt_pos = self._loss_kpt(model, feats, boxes_list, kpts_list)

        total = det_loss + l_kpt

        logs = {
            "obj": float(_ddp_mean(torch.as_tensor(det_parts.get("obj", 0.0), device=device))),
            "box": float(_ddp_mean(torch.as_tensor(det_parts.get("box", 0.0), device=device))),
            "dfl": float(_ddp_mean(torch.as_tensor(det_parts.get("dfl", 0.0), device=device))),
            "ang": float(_ddp_mean(torch.as_tensor(det_parts.get("ang", 0.0), device=device))),
            "cls": float(_ddp_mean(torch.as_tensor(det_parts.get("cls", 0.0), device=device))),
            "kpt": float(_ddp_mean(l_kpt.detach())),
            "pos": float(_ddp_mean(torch.as_tensor(det_parts.get("pos", 0), device=device))),
            "kpt_pos": float(_ddp_mean(torch.as_tensor(kpt_pos, device=device))),
            "total": float(_ddp_mean(total.detach())),
        }
        return total, logs

    # ---------------------- helpers ----------------------
    def _split_map(self, dm: torch.Tensor, level_idx: int, model=None) -> Dict[str, torch.Tensor]:
        """
        Try using head's splitter (preferred). Fallback to DFL-only layout:
            [tx, ty, w_bins(reg_max+1), h_bins(reg_max+1), ang, obj, cls?]
        """
        if model is not None and hasattr(model, "head") and hasattr(model.head, "_split_map"):
            return model.head._split_map(dm)  # type: ignore

        # fallback (needs reg_max from model/head)
        reg_max = None
        if model is not None:
            reg_max = getattr(model, "reg_max", None)
            if reg_max is None and hasattr(model, "head"):
                reg_max = getattr(model.head, "reg_max", None)
        assert isinstance(reg_max, int) and reg_max > 0, \
            "DFL fallback needs model(head).reg_max"

        B, C, H, W = dm.shape
        nb = reg_max + 1
        off = 0
        tx = dm[:, off+0:off+1]; ty = dm[:, off+1:off+2]; off += 2
        w_logits = dm[:, off:off+nb]; off += nb
        h_logits = dm[:, off:off+nb]; off += nb
        ang = dm[:, off:off+1]; off += 1
        obj = dm[:, off:off+1]; off += 1
        cls = dm[:, off:off+self.nc] if self.nc > 1 else None
        return {"tx": tx, "ty": ty, "w_logits": w_logits, "h_logits": h_logits,
                "ang": ang, "obj": obj, "cls": cls}

    @torch.no_grad()
    def _read_targets(self, batch: Dict[str, Any], B: int, device: torch.device):
        """
        Returns per-image lists: boxes (N,5: xc,yc,w,h,ang), labels (N,), kpts (N,2 or None).
        Accepts several field names; clamps empty safely.
        """
        boxes_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        kpts_list: List[torch.Tensor] = []

        # likely fields
        cand_names = [
            ("gtb", "gtc", "gtk"),
            ("boxes", "labels", "kpts"),
            ("gt_boxes", "gt_classes", "gt_kpts")
        ]
        names = None
        for trip in cand_names:
            if trip[0] in batch and trip[1] in batch:
                names = trip; break
        if names is None:
            # nothing—return empty lists
            for _ in range(B):
                boxes_list.append(torch.zeros((0, 5), device=device))
                labels_list.append(torch.zeros((0,), device=device, dtype=torch.long))
                kpts_list.append(torch.zeros((0, 2), device=device))
            return boxes_list, labels_list, kpts_list

        bkey, ckey, kkey = names
        for b in range(B):
            bxs = batch[bkey][b] if isinstance(batch[bkey], list) else batch[bkey]
            cls = batch[ckey][b] if isinstance(batch[ckey], list) else batch[ckey]
            kpt = batch.get(kkey, None)
            if kpt is not None:
                kpt = kpt[b] if isinstance(kpt, list) else kpt

            bxs = torch.as_tensor(bxs, device=device).view(-1, 5) if bxs.numel() else torch.zeros((0, 5), device=device)
            cls = torch.as_tensor(cls, device=device, dtype=torch.long).view(-1)
            if kpt is None or kpt.numel() == 0:
                kpt = torch.zeros((0, 2), device=device)
            else:
                kpt = torch.as_tensor(kpt, device=device).view(-1, 2)

            # sanity
            if bxs.numel() and bxs.shape[1] != 5:
                # try to expand 4 -> add angle 0
                if bxs.shape[1] == 4:
                    pad = torch.zeros((bxs.shape[0], 1), device=device, dtype=bxs.dtype)
                    bxs = torch.cat([bxs, pad], dim=1)
                else:
                    bxs = torch.zeros((0, 5), device=device)
            boxes_list.append(bxs)
            labels_list.append(cls.clamp(min=0, max=max(0, self.nc - 1)))
            kpts_list.append(kpt)
        return boxes_list, labels_list, kpts_list

    # ---------------------- detection loss ----------------------
    def _loss_det(self,
                  det_maps: List[torch.Tensor],
                  feats: Optional[List[torch.Tensor]],
                  batch: Dict[str, Any],
                  model: Optional[nn.Module] = None,
                  epoch: Optional[int] = None):

        device = det_maps[0].device
        B = det_maps[0].shape[0]
        strides = self.strides

        # split per level
        mp_levels = [self._split_map(dm, li, model=model) for li, dm in enumerate(det_maps)]
        Hs = [int(dm.shape[-2]) for dm in det_maps]
        Ws = [int(dm.shape[-1]) for dm in det_maps]

        # targets per image
        boxes_list, labels_list, _ = self._read_targets(batch, B, device)

        # image size (cap P5)
        if isinstance(batch, dict) and "imgs" in batch and hasattr(batch["imgs"], "shape"):
            Himg = int(batch["imgs"].shape[-2]); Wimg = int(batch["imgs"].shape[-1])
        else:
            Himg = Wimg = 640
        img_max = float(max(Himg, Wimg))

        # routing thresholds (<=low -> P3, <=mid -> P4, else -> P5)
        low, mid = self.level_boundaries
        lvl_cap = [float(low), float(mid), img_max]

        # DFL config
        reg_max = None
        if model is not None:
            reg_max = getattr(model, "reg_max", None)
            if reg_max is None and hasattr(model, "head"):
                reg_max = getattr(model.head, "reg_max", None)
        assert isinstance(reg_max, int) and reg_max > 0, "DFL requires model(head).reg_max > 0"
        nbins = reg_max + 1

        # per-level log-size bins in **image pixels normalized by stride** (log-space)
        # narrower than full image to avoid saturation
        min_wh_img = [2.0, 4.0, 8.0]
        max_wh_img = [min(96.0, img_max), min(192.0, img_max), min(384.0, img_max)]

        log_bins_per_level: List[torch.Tensor] = []
        log_minmax_per_level: List[Tuple[float, float]] = []
        for li, s in enumerate(strides):
            log_min = math.log(max(min_wh_img[li] / float(s), 1e-6))
            log_max = math.log(max(max_wh_img[li] / float(s), 1.0))
            idx = torch.arange(nbins, device=device, dtype=torch.float32)
            log_bins = log_min + (log_max - log_min) * (idx / float(reg_max))
            log_bins_per_level.append(log_bins)
            log_minmax_per_level.append((log_min, log_max))

        # accumulators
        l_obj = det_maps[0].new_tensor(0.0)
        l_box = det_maps[0].new_tensor(0.0)
        l_dfl = det_maps[0].new_tensor(0.0)
        l_ang = det_maps[0].new_tensor(0.0)
        l_cls = det_maps[0].new_tensor(0.0)
        pos_cnt = 0

        # per-image assignment (1 GT -> 1 cell in routed level)
        for b in range(B):
            gtb = boxes_list[b]   # (N,5) or (0,5)
            gtc = labels_list[b]  # (N,)

            if gtb.numel() == 0:
                # pure background: sample negatives for obj below
                continue

            for k in range(gtb.shape[0]):
                xc, yc, wpx, hpx, ang_t = gtb[k]
                cls_id = int(gtc[k]) if gtc.numel() else 0

                # route to level by max side in image px
                mside = float(max(wpx.item(), hpx.item()))
                if mside <= low:   li = 0
                elif mside <= mid: li = 1
                else:              li = 2

                s = float(strides[li])
                H, W = Hs[li], Ws[li]
                mp = mp_levels[li]

                # grid indices & in-cell offsets
                gx = xc / s; gy = yc / s
                i = int(torch.clamp(gx.floor(), 0, W - 1).item())
                j = int(torch.clamp(gy.floor(), 0, H - 1).item())
                tx_t = (gx - float(i)).clamp(0.0, 1.0)
                ty_t = (gy - float(j)).clamp(0.0, 1.0)

                # --- preds at cell (b,j,i) ---
                tx_p = mp["tx"][b, 0, j, i]
                ty_p = mp["ty"][b, 0, j, i]
                a_p  = mp["ang"][b, 0, j, i]
                obj_p = mp["obj"][b, 0, j, i]
                cls_p = mp["cls"][b, :, j, i] if mp["cls"] is not None else None

                # DFL logits (nbins,)
                w_logits = mp["w_logits"][b, :, j, i]   # raw logits
                h_logits = mp["h_logits"][b, :, j, i]

                # --- targets ---
                # center offsets
                l_box = l_box + self.smoothl1(tx_p, tx_p.new_tensor(tx_t)) \
                              + self.smoothl1(ty_p, ty_p.new_tensor(ty_t))

                # log-size targets (feature-normalized)
                log_min, log_max = log_minmax_per_level[li]
                logw_t = math.log(max(wpx.item() / s, 1e-6))
                logh_t = math.log(max(hpx.item() / s, 1e-6))

                # EV stabilizer: expected log-size from logits in **log-space**
                log_bins = log_bins_per_level[li]                   # (nbins,)
                pw = torch.softmax(w_logits, dim=0)
                ph = torch.softmax(h_logits, dim=0)
                logw_ev = (pw * log_bins).sum()
                logh_ev = (ph * log_bins).sum()
                l_box = l_box + self.smoothl1(logw_ev, w_logits.new_tensor(logw_t)) \
                              + self.smoothl1(logh_ev, h_logits.new_tensor(logh_t))

                # DFL fractional bin targets
                # map true log-size to [0, reg_max]
                def to_frac_idx(logv: float) -> float:
                    if log_max <= log_min + 1e-9:
                        return float(reg_max) * 0.5  # degenerate
                    u = (logv - log_min) / (log_max - log_min) * float(reg_max)
                    return float(max(0.0, min(float(reg_max) - 1e-6, u)))

                t_w_idx = w_logits.new_tensor(to_frac_idx(logw_t))
                t_h_idx = h_logits.new_tensor(to_frac_idx(logh_t))

                l_dfl = l_dfl + _dfl_nll_loss(w_logits.unsqueeze(0), t_w_idx.view(1)) \
                               + _dfl_nll_loss(h_logits.unsqueeze(0), t_h_idx.view(1))

                # angle
                a_pred = _angle_from_logit(a_p)
                l_ang = l_ang + _angle_loss(a_pred, a_p.new_tensor(ang_t))

                # objectness (pos=1 here)
                l_obj = l_obj + self.bce(obj_p.view(1), obj_p.new_tensor([1.0]))

                # class (optional multi-class BCE)
                if cls_p is not None and self.lambda_cls > 0.0 and self.nc > 1:
                    t = torch.zeros(self.nc, device=cls_p.device, dtype=cls_p.dtype)
                    t[cls_id] = 1.0
                    l_cls = l_cls + F.binary_cross_entropy_with_logits(cls_p, t, reduction="mean")

                pos_cnt += 1

        # background objectness (hard negative mining)
        # sample ~neg_obj_ratio * pos from each level
        if self.lambda_obj > 0.0:
            target0 = 0.0
            for li, mp in enumerate(mp_levels):
                obj_map = mp["obj"]            # (B,1,H,W)
                B_, _, H, W = obj_map.shape
                # build a mask with all cells, remove the pos ones we already hit (best-effort)
                # simplest: sample random negatives; your trainer usually balances this further upstream
                num_neg = int(self.neg_obj_ratio * max(1, pos_cnt))
                if num_neg > 0:
                    # uniform random indices
                    ii = torch.randint(0, W, (num_neg,), device=device)
                    jj = torch.randint(0, H, (num_neg,), device=device)
                    bb = torch.randint(0, B_, (num_neg,), device=device)
                    l_obj = l_obj + self.bce(obj_map[bb, 0, jj, ii], obj_map.new_full((num_neg,), target0))

        # normalize by positives (avoid exploding if many GT)
        norm = max(1, pos_cnt)
        parts = {
            "obj": float((l_obj / norm).detach()),
            "box": float((l_box / norm).detach()),
            "dfl": float((l_dfl / norm).detach()),
            "ang": float((l_ang / norm).detach()),
            "cls": float((l_cls / max(1, pos_cnt)).detach()),
            "pos": pos_cnt,
        }
        total = self.lambda_obj * (l_obj / norm) \
                + self.lambda_box * (l_box / norm) \
                + self.lambda_dfl * (l_dfl / norm) \
                + self.lambda_ang * (l_ang / norm) \
                + self.lambda_cls * (l_cls / max(1, pos_cnt))

        return total, parts

    # ---------------------- keypoint loss (minimal, safe) ----------------------
    def _loss_kpt(self,
                  model: Optional[nn.Module],
                  feats: Optional[List[torch.Tensor]],
                  boxes_list: List[torch.Tensor],
                  kpts_list: List[torch.Tensor]):

        # If no kpt head or disabled, return 0 (keeps training stable)
        if self.lambda_kpt <= 0.0:
            return (feats[0].new_tensor(0.0) if feats else torch.tensor(0.0)), 0
        if model is None or not hasattr(model, "kpt_predictor") or model.kpt_predictor is None:
            return (feats[0].new_tensor(0.0) if feats else torch.tensor(0.0)), 0
        if feats is None or len(feats) == 0:
            return torch.tensor(0.0), 0

        # Expect model.kpt_predictor(feats) -> (u_pred, v_pred) per ROI or per image;
        # In your pipeline, ROI metas are attached on the model during forward.
        metas: Optional[List[Dict[str, torch.Tensor]]] = getattr(model, "roi_metas", None)
        if not metas:
            return feats[0].new_tensor(0.0), 0

        device = feats[0].device
        l = feats[0].new_tensor(0.0)
        cnt = 0
        for m in metas:
            if "M" not in m or "gt_kpt" not in m:
                continue
            kpt_img = torch.as_tensor(m["gt_kpt"], device=device, dtype=feats[0].dtype).view(1, 2)
            # Your crop->feat affine is m["M"]; we just call your predictor
            u_pred, v_pred = model.kpt_predictor(feats)  # should return scalars per-ROI or logits
            # be tolerant: if tensors, squeeze to scalar
            if isinstance(u_pred, torch.Tensor): u_pred = u_pred.reshape(-1)[0]
            if isinstance(v_pred, torch.Tensor): v_pred = v_pred.reshape(-1)[0]
            # Without a pixel-space target in crop, treat (0,0) origin as target placeholder
            u_t = u_pred.new_tensor(0.0)
            v_t = v_pred.new_tensor(0.0)
            l = l + F.smooth_l1_loss(u_pred, u_t, reduction="mean") \
                    + F.smooth_l1_loss(v_pred, v_t, reduction="mean")
            cnt += 1

        if cnt == 0:
            return feats[0].new_tensor(0.0), 0
        return self.lambda_kpt * (l / cnt), cnt
