# -*- coding: utf-8 -*-
"""
YOLO11-OBB-Pose (two-stage keypoint on crops) — DFL-only

- Backbone -> P3,P4,P5
- PAN-FPN neck -> (N3,D4,D5) all with the same channel (neck.ch_out)
- OBBPoseHead -> detection maps at 3 scales
- kpt_from_obbs(feats, boxes_list, ...) -> rotated crops on P3, tiny local head -> (u,v) in [0,1]

Notes:
- This file enforces DFL-only for (w,h) (and optionally (cx,cy) if your head supports it).
- Angle is single-logit (YOLO-style) on the head side; no classic sin/cos anywhere here.
- We set .strides on the head and pass DFL flags to avoid compatibility surprises.
- The decoder wrapper tolerates various kwarg names and fails safe to empty predictions.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn

# ---- use your existing modules ----
# backbone must return (p3,p4,p5) with channels like (256,512,1024)
from .backbones.cspnext11 import CSPBackbone
from .necks.pan_fpn import PANFPN
from .heads.obbpose_head import OBBPoseHead
from .layers.rot_roi import RotatedROIPool


# -------------------------
# Tiny local crop->(u,v) head
# -------------------------
class _CropKptHead(nn.Module):
    """
    Minimal crop -> (u,v) in [0,1]. Global pooling, so independent of crop size S.
    Lazily instantiated inside the main model using the observed P3 channels.
    """
    def __init__(self, in_ch: int, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, S, S) -> (N, 2) constrained to [0,1]
        y = self.conv(x).flatten(1)  # (N, hidden)
        y = self.fc(y)               # (N, 2)
        return y.sigmoid()


# -------------------------
# Helper: try-call with flexible kwargs, fall back safely
# -------------------------
def _safe_decode(callable_fn, *args, **kwargs):
    """Attempt decode with tolerant kwargs; progressively relax on TypeError."""
    # common alias normalization
    if "score_thr" in kwargs and "conf_thres" not in kwargs:
        kwargs["conf_thres"] = kwargs.pop("score_thr")
    if "iou_thr" in kwargs and "iou_thres" not in kwargs:
        kwargs["iou_thres"] = kwargs.pop("iou_thr")
    if "max_det" not in kwargs and "max_det_per_img" in kwargs:
        kwargs["max_det"] = kwargs.pop("max_det_per_img")

    # first try with all kwargs
    try:
        return callable_fn(*args, **kwargs)
    except TypeError:
        pass

    # try a pared-down set that most decoders support
    slim = {}
    for k in ("conf_thres", "iou_thres", "max_det", "use_nms"):
        if k in kwargs:
            slim[k] = kwargs[k]
    try:
        return callable_fn(*args, **slim)
    except TypeError:
        pass

    # last resort: no kwargs
    return callable_fn(*args)


# -------------------------
# Main model
# -------------------------
class YOLO11_OBBPOSE_TD(nn.Module):
    def __init__(
        self,
        num_classes: int,
        width: float = 1.0,
        backbone: Optional[nn.Module] = None,
        neck: Optional[PANFPN] = None,
        head: Optional[OBBPoseHead] = None,
        feat_down: int = 8,            # stride of P3
        kpt_head_hidden: int = 128,    # hidden width for crop head
        # DFL config
        reg_max: int = 16,
        strides: Tuple[int, int, int] = (8, 16, 32),
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.width = float(width)
        self._schema_tag = "explicit-pan v2 (DFL-only)"
        self._warned_decode_fallback = False

        # ---- Backbone ----
        self.backbone: nn.Module = backbone if backbone is not None else CSPBackbone(width=self.width)

        # Try to get backbone out channels (for sanity prints)
        bch = getattr(self.backbone, "out_channels", (256, 512, 1024))
        if isinstance(bch, (list, tuple)) and len(bch) == 3:
            c3, c4, c5 = [int(x) for x in bch]
        else:
            c3, c4, c5 = 256, 512, 1024

        # ---- Neck: PANFPN ----
        self.neck: PANFPN = neck if neck is not None else PANFPN(ch_in=(c3, c4, c5), ch_out=256)
        self.neck_ch_out: int = int(getattr(self.neck, "ch_out", 256))

        # ---- Detection Head (DFL-only) ----
        if head is None:
            # Pass safest, widely-supported args; set attributes after init for the rest.
            head = OBBPoseHead(
                ch=(self.neck_ch_out,)*3,
                num_classes=self.num_classes,
                strides=strides,           # many heads require this
            )
        # Enforce DFL-only behavior on the head regardless of external defaults
        # We set attributes rather than assuming constructor kwargs exist.
        head.use_dfl = True
        head.angle_single_logit = True
        head.use_classic = False           # if the head supports it
        head.reg_max = int(reg_max)        # number of DFL bins - 1
        head.strides = tuple(int(s) for s in strides)

        # Optional: provide reasonable default DFL log ranges on each level (safe if unused)
        try:
            # log(min/stride) .. log(max/stride); keep modest to avoid collapse on small datasets
            # Here we cover approx [1px..128px] per level by default.
            head.dfl_log_minmax = (
                (math.log(2 / 8), math.log(96 / 8)),
                (math.log(2 / 16), math.log(96 / 16)),
                (math.log(2 / 32), math.log(96 / 32)),
            )
        except Exception:
            pass

        self.head: OBBPoseHead = head

        # ---- ROI layer for rotated crops on P3 ----
        self.roi = RotatedROIPool(feat_down=feat_down)

        # ---- Lazy local keypoint head (created on first use to match P3 channels/device) ----
        self._kpt_head_hidden = int(kpt_head_hidden)
        self.kpt_head: Optional[nn.Module] = None
        self._kpt_head_in_ch: Optional[int] = None

        # ---- Debug print to match your logs ----
        params = sum(p.numel() for p in self.parameters())
        print(f"[SCHEMA TRAIN] sig={self._schema_tag} params={params}", flush=True)
        print(f"[SCHEMA TRAIN] backbone_out=({c3},{c4},{c5}) neck_ch_out={self.neck_ch_out}", flush=True)

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Returns:
            det_maps: list of 3 detection maps (P3,P4,P5)
            feats   : list of 3 features    (P3,P4,P5 from the neck)
        """
        # Backbone
        p3, p4, p5 = self.backbone(x)

        # Neck
        n3, d4, d5 = self.neck(p3, p4, p5)  # all with channel = neck_ch_out

        # Head
        det_maps, _ = self.head([n3, d4, d5])  # OBBPoseHead returns (det, kpt); only det is used here

        # Return maps + features for the loss
        return det_maps, [n3, d4, d5]

    # -------------------------
    # Decoder (DFL-only path, tolerant to kwargs)
    # -------------------------
    @torch.no_grad()
    def decode_obb_from_pyramids(self, det_maps: List[torch.Tensor], imgs: torch.Tensor, **kwargs):
        """
        Returns per-image predictions expected by your Evaluator:
            List[Dict[str, Tensor]]:
                - 'boxes':  (N,5) tensor  (cx, cy, w, h, angle_radians)
                - 'scores': (N,)  tensor
                - 'labels': (N,)  tensor (int64)
        """
        # prefer head's decoder if present
        if hasattr(self.head, "decode_obb_from_pyramids") and callable(self.head.decode_obb_from_pyramids):
            try:
                return _safe_decode(self.head.decode_obb_from_pyramids, det_maps, imgs, **kwargs)
            except Exception as e:
                if not self._warned_decode_fallback:
                    print(f"[decode] WARNING head.decode failed: {e}", flush=True)
                    self._warned_decode_fallback = True

        # generic fallbacks on the head/model (name variations)
        for mod in (self.head, self):
            if mod is None:
                continue
            for cand in ("decode_from_pyramids", "decode_obb_pyramids", "decode_obb",
                         "decode", "postprocess", "predict"):
                fn = getattr(mod, cand, None)
                if callable(fn):
                    try:
                        return _safe_decode(fn, det_maps, imgs, **kwargs)
                    except Exception as e:
                        if not self._warned_decode_fallback:
                            print(f"[decode] WARNING {cand} failed: {e}", flush=True)
                            self._warned_decode_fallback = True

        # Safe EMPTY fallback — do not raise; lets training/eval proceed
        B = int(imgs.shape[0]) if imgs is not None and hasattr(imgs, "shape") else 1
        device = imgs.device if isinstance(imgs, torch.Tensor) else next(self.parameters()).device
        empty = lambda: {
            "boxes":  torch.zeros((0, 5), device=device, dtype=torch.float32),
            "scores": torch.zeros((0,), device=device, dtype=torch.float32),
            "labels": torch.zeros((0,), device=device, dtype=torch.long),
        }
        if not self._warned_decode_fallback:
            print("[decode] WARNING: using EMPTY fallback (DFL-only model; no head decoder matched).",
                  flush=True)
            self._warned_decode_fallback = True
        return [empty() for _ in range(B)]

    # -------------------------
    # Two-stage keypoint from OBBs -> crops -> (u,v)
    # -------------------------
    @torch.no_grad()
    def kpt_from_obbs(
            self,
            feats: List[torch.Tensor],                      # [P3,P4,P5]
            boxes_list: List[Optional[torch.Tensor]],       # per-image OBBs: (Ni,5)
            scores_list: Optional[List[Optional[torch.Tensor]]] = None,
            kpt_topk: int = 256,
            kpt_crop: int = 3,
            kpt_min_per_img: int = 2,
            chunk: int = 512,
            **kwargs,  # accept legacy arg names without breaking
    ):
        """
        Returns:
            uv_pred: (M,2) in [0,1]
            metas  : list of dicts (one per crop) with bbox + geometry + bix (image index)
        """
        # aliases
        if "topk" in kwargs:        kpt_topk = int(kwargs["topk"])
        if "crop" in kwargs:        kpt_crop = int(kwargs["crop"])
        if "min_per_img" in kwargs: kpt_min_per_img = int(kwargs["min_per_img"])
        if "chunks" in kwargs:      chunk = int(kwargs["chunks"])
        if "chunk_size" in kwargs:  chunk = int(kwargs["chunk_size"])

        def _to_2d_tensor(x, device, dtype):
            t = torch.as_tensor(x, device=device, dtype=dtype)
            if t.ndim == 1:
                if t.numel() == 0:
                    return t.new_zeros((0, 5))
                t = t.unsqueeze(0)
            if t.ndim != 2 or t.shape[-1] != 5:
                raise ValueError(f"OBB tensor must be (N,5); got {tuple(t.shape)}")
            return t

        device = feats[0].device
        P3 = feats[0]
        B = int(P3.shape[0])

        # wire crop size on ROI
        if hasattr(self.roi, "S"):
            self.roi.S = int(kpt_crop)
        elif hasattr(self.roi, "crop_size"):
            self.roi.crop_size = int(kpt_crop)

        # lazy-build the crop head to match P3 channels/device
        in_ch = int(P3.shape[1])
        if (self.kpt_head is None) or (getattr(self, "_kpt_head_in_ch", None) != in_ch):
            self.kpt_head = _CropKptHead(in_ch=in_ch, hidden=self._kpt_head_hidden).to(device).float()
            self._kpt_head_in_ch = in_ch

        uv_chunks: List[torch.Tensor] = []
        meta_all: List[dict] = []

        if len(boxes_list) != B:
            raise ValueError(f"boxes_list length {len(boxes_list)} must equal batch size {B}")

        for bix in range(B):
            rois = boxes_list[bix]
            if rois is None:
                continue

            rois = _to_2d_tensor(rois, device=device, dtype=P3.dtype)
            if rois.numel() == 0:
                continue

            # optional top-k
            if scores_list is not None and isinstance(scores_list[bix], torch.Tensor):
                sc = scores_list[bix].to(device=device)
                if sc.numel() and sc.numel() == rois.shape[0]:
                    k = min(int(kpt_topk), sc.numel())
                    idx = torch.topk(sc, k=k, largest=True, sorted=False).indices
                    rois = rois[idx]
            else:
                if rois.shape[0] > kpt_topk:
                    rois = rois[:kpt_topk]

            # ensure minimum crops
            if rois.shape[0] < kpt_min_per_img and rois.shape[0] > 0:
                rep = kpt_min_per_img - rois.shape[0]
                rois = torch.cat([rois, rois[:rep]], dim=0)

            # chunked ROI inference
            N = int(rois.shape[0])
            s = 0
            while s < N:
                e = min(s + int(chunk), N)
                rois_slice = rois[s:e]
                if rois_slice.ndim == 1:
                    rois_slice = rois_slice.unsqueeze(0)

                crops, metas = self.roi(P3[bix:bix + 1], rois_slice)  # (M,C,S,S), list[dict]

                head_dtype = next(self.kpt_head.parameters()).dtype
                if crops.dtype != head_dtype:
                    crops = crops.to(head_dtype)

                uv = self.kpt_head(crops)  # (M,2) in [0,1]

                for m in metas:
                    m["bix"] = int(bix)

                uv_chunks.append(uv)
                meta_all.extend(metas)
                s = e

        if not uv_chunks:
            return P3.new_zeros((0, 2)), []

        return torch.cat(uv_chunks, dim=0), meta_all
