# -*- coding: utf-8 -*-
"""
YOLO11-OBB-Pose (two-stage keypoint on crops)

- Backbone -> P3,P4,P5
- PAN-FPN neck -> (N3,D4,D5) all with the same channel (neck.ch_out)
- OBBPoseHead -> detection maps at 3 scales
- kpt_from_obbs(feats, boxes_list, ...) -> rotated crops on P3, tiny local head -> (u,v) in [0,1]

This file intentionally avoids introducing new global modules; it only defines a
minimal local crop head for keypoint regression that is lazily created to match
P3 channels and device. Everything else uses your existing backbone / PANFPN /
OBBPoseHead / RotatedROI.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

# ---- use your existing modules ----
# backbone must return (p3,p4,p5) with channels like (256,512,1024)
from .backbones.cspnext11 import CSPBackbone  # adapt import if your symbol differs
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
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.width = float(width)
        self._schema_tag = "explicit-pan v2"

        # ---- Backbone ----
        # Expecting backbone that outputs tuple(P3,P4,P5) where channels are roughly (256,512,1024)
        self.backbone: nn.Module = backbone if backbone is not None else CSPBackbone(width=self.width)

        # Try to get backbone out channels (for sanity prints)
        bch = getattr(self.backbone, "out_channels", (256, 512, 1024))
        if isinstance(bch, (list, tuple)) and len(bch) == 3:
            c3, c4, c5 = [int(x) for x in bch]
        else:
            # Fallback if backbone doesn't expose out_channels; use typical values
            c3, c4, c5 = 256, 512, 1024

        # ---- Neck: PANFPN ----
        # Make sure PANFPN lateral convs match backbone channels
        self.neck: PANFPN = neck if neck is not None else PANFPN(ch_in=(c3, c4, c5), ch_out=256)

        # For heads that want to know P3 channels
        self.neck_ch_out: int = int(getattr(self.neck, "ch_out", 256))

        # ---- Detection Head (kept as in your repo) ----
        # If your OBBPoseHead expects a tuple of channels per scale, pass (neck_ch_out,)*3.
        self.head: OBBPoseHead = head if head is not None else OBBPoseHead(
            ch=(self.neck_ch_out,)*3,
            num_classes=self.num_classes,
        )

        # ---- ROI layer for rotated crops on P3 ----
        # We avoid fixing crop size here; we’ll set it per-call from kpt_from_obbs(kpt_crop=...)
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
        p3, p4, p5 = self.backbone(x)      # shapes like: (B,256,80,80), (B,512,40,40), (B,1024,20,20)

        # Neck
        n3, d4, d5 = self.neck(p3, p4, p5) # all with channel = neck_ch_out

        # Head
        det_maps, _ = self.head([n3, d4, d5])  # OBBPoseHead returns (det, kpt); only det is used here

        # Return maps + features for the loss
        return det_maps, [n3, d4, d5]

    # -------------------------
    # Two-stage keypoint from OBBs -> crops -> (u,v)
    # -------------------------
    @torch.no_grad()
    def kpt_from_obbs(
            self,
            feats,  # [P3,P4,P5]
            boxes_list,  # per-image OBBs: (Ni,5) [cx,cy,w,h,deg], or list/np
            scores_list=None,
            kpt_topk: int = 256,
            kpt_crop: int = 3,
            kpt_min_per_img: int = 2,
            kpt_fallback_frac: float = 0.4,
            chunk: int = 512,
            **kwargs,  # accept legacy names
    ):
        """
        Returns:
            uv_pred: (M,2) in [0,1]
            metas  : list of dicts (one per crop) with bbox + geometry + bix (image index)
        """
        # ---- alias mapping (compat with caller) ----
        if "topk" in kwargs:            kpt_topk = int(kwargs["topk"])
        if "crop" in kwargs:            kpt_crop = int(kwargs["crop"])
        if "min_per_img" in kwargs:     kpt_min_per_img = int(kwargs["min_per_img"])
        if "fallback_frac" in kwargs:   kpt_fallback_frac = float(kwargs["fallback_frac"])
        if "chunks" in kwargs:          chunk = int(kwargs["chunks"])
        if "chunk_size" in kwargs:      chunk = int(kwargs["chunk_size"])

        def _to_2d_tensor(x, device, dtype):
            """
            Normalize inputs to strict (N,5) float tensor on the right device/dtype.
            - empty -> (0,5)
            - single ROI like (5,) -> (1,5)
            """
            t = torch.as_tensor(x, device=device, dtype=dtype)
            if t.ndim == 1:
                if t.numel() == 0:
                    return t.new_zeros((0, 5))
                t = t.unsqueeze(0)  # (1,5)
            if t.ndim != 2:
                raise ValueError(f"OBB tensor must be 2-D (N,5); got {tuple(t.shape)}")
            if t.shape[-1] != 5:
                raise ValueError(f"OBB tensor last dim must be 5; got {tuple(t.shape)}")
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

            # ---- normalize to (N,5) on correct device/dtype
            rois = _to_2d_tensor(rois, device=device, dtype=P3.dtype)
            if rois.numel() == 0:
                continue

            # (optional) score-based per-image top-k
            if scores_list is not None and isinstance(scores_list[bix], torch.Tensor):
                sc = scores_list[bix].to(device=device)
                if sc.numel() and sc.numel() == rois.shape[0]:
                    k = min(int(kpt_topk), sc.numel())
                    idx = torch.topk(sc, k=k, largest=True, sorted=False).indices
                    rois = rois[idx]
            else:
                if rois.shape[0] > kpt_topk:
                    rois = rois[:kpt_topk]

            # ensure at least kpt_min_per_img crops unless empty
            if rois.shape[0] < kpt_min_per_img and rois.shape[0] > 0:
                # simple fallback: repeat first few boxes
                rep = kpt_min_per_img - rois.shape[0]
                rois = torch.cat([rois, rois[:rep]], dim=0)

            # chunking to limit memory
            N = int(rois.shape[0])
            s = 0
            while s < N:
                e = min(s + int(chunk), N)
                rois_slice = rois[s:e]
                # ---- STRICT: keep 2-D even for single element slices
                if rois_slice.ndim == 1:
                    rois_slice = rois_slice.unsqueeze(0)

                # ROI crops on the single-image feature map
                crops, metas = self.roi(P3[bix:bix + 1], rois_slice)  # (M,C,S,S), list[dict]
                # keep params dtype as source of truth
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
