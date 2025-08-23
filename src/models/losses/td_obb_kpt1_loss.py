# src/models/losses/td_obb_kpt1_loss.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- small utils -------------------------

def _dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def _ddp_mean(x: torch.Tensor) -> torch.Tensor:
    if not _dist_is_initialized():
        return x
    y = x.clone()
    torch.distributed.all_reduce(y, op=torch.distributed.ReduceOp.SUM)
    y /= torch.distributed.get_world_size()
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

        # -------- parse GT (supports two formats) --------
        boxes_list, labels_list, kpts_list = self._read_targets(batch, B, device)

        # -------- detection loss (obj, box, angle, cls) --------
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
        model: Optional[nn.Module],
        feats: Optional[List[torch.Tensor]],
        boxes_list: List[torch.Tensor],
        kpts_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, int]:
        """
        Top-down 1-keypoint loss on P3 crops.

        Assumptions:
          - model.kpt_from_obbs(feats, boxes_list, scores_list=None) -> (uv_pred, metas)
          - metas carry per-ROI affine 'M' (2x3, crop->feat) and model.roi.feat_down
          - kpts_list[i] aligns with boxes_list[i] within image i (same ordering)
        """
        # Guard rails
        if model is None or feats is None or len(feats) == 0:
            # No features / no model: no keypoint supervision
            dev = boxes_list[0].device if len(boxes_list) else torch.device("cpu")
            return torch.zeros((), device=dev), 0

        device = feats[0].device
        dtype = feats[0].dtype

        uv_losses: List[torch.Tensor] = []
        n_pos = 0

        # Helper: build UV target in crop pixels from image pixels using ROI meta
        def _uv_from_meta(kpt_xy_img: torch.Tensor, M_crop_to_feat: torch.Tensor, feat_down: float) -> torch.Tensor:
            # image px -> feature px
            xy_feat = kpt_xy_img.to(device=device, dtype=dtype) / float(feat_down)
            # invert 2x3
            A = M_crop_to_feat[:, :2]
            t = M_crop_to_feat[:, 2:]
            det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
            # safe inverse
            invA00 = torch.where(det.abs() > 1e-12,  A[:, 1, 1] / det, torch.ones_like(det))
            invA01 = torch.where(det.abs() > 1e-12, -A[:, 0, 1] / det, torch.zeros_like(det))
            invA10 = torch.where(det.abs() > 1e-12, -A[:, 1, 0] / det, torch.zeros_like(det))
            invA11 = torch.where(det.abs() > 1e-12,  A[:, 0, 0] / det, torch.ones_like(det))
            invA = torch.stack([torch.stack([invA00, invA01], dim=-1),
                                torch.stack([invA10, invA11], dim=-1)], dim=1)  # (N,2,2)
            invt = -(invA @ t)  # (N,2,1)
            uv = (invA @ xy_feat.unsqueeze(-1) + invt).squeeze(-1)  # (N,2)
            return uv

        # Iterate per image to avoid the earlier batch/length mismatch
        for bix, (boxes, kpts) in enumerate(zip(boxes_list, kpts_list)):
            # Normalize shapes to (N,5) and (N,2)
            if boxes.numel() == 0 or kpts.numel() == 0:
                continue
            boxes = boxes.reshape(-1, 5).to(device=device, dtype=dtype)
            kpts = kpts.reshape(-1, 2).to(device=device, dtype=dtype)

            # Call model for *this image only* (singleton list to match its API)
            with torch.cuda.amp.autocast(enabled=False):
                uv_pred, metas = model.kpt_from_obbs(feats, [boxes], scores_list=None)

            if uv_pred is None or (isinstance(uv_pred, torch.Tensor) and uv_pred.numel() == 0):
                continue

            # Unwrap metas to a flat list[dict] aligned with the input OBBs
            metas_img = metas
            if isinstance(metas_img, (list, tuple)) and len(metas_img) == 1:
                metas_img = metas_img[0]
            if isinstance(metas_img, dict) and "list" in metas_img:
                metas_img = metas_img["list"]
            assert isinstance(metas_img, (list, tuple)), "ROI metas must be a list aligned with OBBs."

            # Trim to the common length just in case
            N = min(len(metas_img), boxes.shape[0], kpts.shape[0])
            if N == 0:
                continue
            metas_img = metas_img[:N]
            kpts_img = kpts[:N]

            # Collect affine matrices and build uv targets
            Ms = []
            for m in metas_img:
                M = m["M"]
                M = torch.as_tensor(M, device=device, dtype=dtype)
                if M.ndim == 2:
                    M = M.unsqueeze(0)  # (1,2,3)
                Ms.append(M)
            M_cat = torch.cat(Ms, dim=0)  # (N,2,3)

            # feat_down from the ROI module (defaults to 8 if not present)
            feat_down = float(getattr(getattr(model, "roi", None), "feat_down", 8.0))
            uv_tgt = _uv_from_meta(kpts_img, M_cat, feat_down)  # (N,2)

            # Ensure predictions line up with N
            if isinstance(uv_pred, (list, tuple)):
                # some heads may return list per image
                uv_pred = uv_pred[0]
            uv_pred = torch.as_tensor(uv_pred, device=device).to(dtype=uv_tgt.dtype)
            uv_pred = uv_pred.reshape(-1, 2)[:N]

            # Smooth L1 in crop pixels
            uv_losses.append(F.smooth_l1_loss(uv_pred, uv_tgt, reduction="mean"))
            n_pos += int(N)

        if not uv_losses:
            return torch.zeros((), device=device, dtype=dtype), 0

        return torch.stack(uv_losses).mean(), n_pos
