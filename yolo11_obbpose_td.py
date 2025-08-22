
# Copyright (c) 2025.
# YOLO11 OBB+Keypoint (two-stage TD) model definition.
# Notes:
# - Builds PAN-FPN immediately (no Lazy wrapper). We infer backbone P3/P4/P5 channels
#   with a tiny dry-run at __init__ so head channels match and DDP sees a fixed graph.
# - Public methods preserved: forward(), decode(), kpt_from_obbs(), num_params(), schema_signature().

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Backbone ---
# Expect a CSPNeXt11-style backbone that returns (P3,P4,P5) feature maps.
# Your repo provides this in: src/models/backbones/cspnext11.py
try:
    from src.models.backbones.cspnext11 import CSPNeXt11 as CSPBackbone
except Exception:
    # Fallback alias (some repos name it cspnext11.Backbone)
    from src.models.backbones.cspnext11 import CSPBackbone as CSPBackbone  # type: ignore


# --- Neck ---
from src.models.necks.pan_fpn import PANFPN


# --- Head (detections & per-level keypoint heatmaps/logits) ---
from src.models.heads.obbpose_head import OBBPoseHead


# --- Rotated ROI crop for TD keypoint head ---
from src.models.layers.rot_roi import RotatedROIPool  # provides (crops, metas)


# === Small helpers ============================================================

def _num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _schema_signature(model: nn.Module) -> str:
    # Short digest of parameter shapes; stable across ranks
    h = 0x12345ABC
    for k, v in model.state_dict().items():
        # xor-in sizes and a simple hash of the name
        h ^= len(k) * 1315423911
        for d in v.shape:
            h ^= (int(d) + 0x9e3779b97f4a7c15) & 0xFFFFFFFF
        h &= 0xFFFFFFFF
    return f"{h:08x}"


def _src_path(obj: Any) -> str:
    try:
        import inspect, os
        return inspect.getsourcefile(obj.__class__) or "<?>"
    except Exception:
        return "<?>"


# === TD keypoint refinement head (simple conv head over cropped P3 features) ===
class KptTDHead(nn.Module):
    def __init__(self, C: int = 256, S: int = 3):
        super().__init__()
        self.S = int(S)
        self.conv = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.SiLU(inplace=True),
            nn.Conv2d(C, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.SiLU(inplace=True),
        )
        self.fc = nn.Linear(C * S * S, 2)  # predict (u,v) in crop coords

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, S, S) -> (N, 2) (u,v) in [0,S)
        y = self.conv(x)
        y = y.flatten(1)
        y = self.fc(y)
        return y


# === Main Model ===============================================================
class YOLO11_OBBPOSE_TD(nn.Module):
    """
    Returns
    -------
    (det_maps, feats):
        det_maps: List[Tensor] of length 3, each (B, A, H, W, K) or (B, K, H, W) depending on head
        feats:    List[Tensor] [P3, P4, P5] (B, C, H, W) from neck; used by loss/ROI
    """

    def __init__(
        self,
        num_classes: int,
        width: float = 1.0,
        depth: float = 1.0,
        kpt_crop: int = 3,
        fpn_out: int = 256,
        backbone: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.nc = int(num_classes)
        self.width = float(width)
        self.depth = float(depth)
        self.kpt_crop = int(kpt_crop)
        self.fpn_out = int(fpn_out)

        # --- Backbone ---
        self.backbone = backbone if backbone is not None else CSPBackbone(in_ch=3, width=self.width, depth=self.depth)

        # --- Infer P3/P4/P5 channels immediately (CPU tiny dry-run) ---
        with torch.no_grad():
            self.backbone.eval()
            dummy = torch.zeros(1, 3, 256, 256)  # small to avoid OOM
            p3, p4, p5 = self.backbone(dummy)
            c3, c4, c5 = p3.shape[1], p4.shape[1], p5.shape[1]
        # (No need to keep backbone in eval; trainer will switch modes as needed)
        self.backbone.train()

        # --- Neck: build PAN-FPN now, with fixed out channels ---
        if neck is not None:
            self.neck = neck
            # optional: ensure it provides .ch / .ch_out
            if not hasattr(self.neck, "ch"):
                # assume neck preserves out channels per level = fpn_out
                self.neck.ch = (self.fpn_out, self.fpn_out, self.fpn_out)  # type: ignore[attr-defined]
        else:
            self.neck = PANFPN(ch=(c3, c4, c5), out_ch=self.fpn_out)

        # --- Head (det/kpt heatmaps per level) ---
        self.head: OBBPoseHead = head if head is not None else OBBPoseHead(ch=self.neck.ch, nc=self.nc)  # type: ignore[arg-type]

        # --- TD keypoint refinement (ROI over P3 features) ---
        self.roi = RotatedROIPool(out_size=self.kpt_crop)  # returns (crops, metas) given P3 and per-image OBBs
        self.kpt_td = KptTDHead(C=self.neck.ch[0], S=self.kpt_crop)

        # --- Friendly schema logs ---
        try:
            sig = _schema_signature(self)
            tot = _num_params(self)
            print(f"[SCHEMA TRAIN] sig={sig} params={tot}", flush=True)
            print(f"[SCHEMA TRAIN] backbone from: {_src_path(self.backbone)}", flush=True)
            print(f"[SCHEMA TRAIN] neck from: {_src_path(self.neck)}", flush=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        images: (B,3,H,W)
        Returns: (det_maps, feats)
            det_maps: [D3, D4, D5] tensors per head spec
            feats:    [N3, N4, N5] feature maps (after PANFPN)
        """
        p3, p4, p5 = self.backbone(images)        # /8, /16, /32
        n3, n4, n5 = self.neck(p3, p4, p5)        # unified channels = self.fpn_out
        det_maps, kpt_maps = self.head([n3, n4, n5])
        # Criterion uses feats; kpt_maps are kept inside head (not returned) or can be returned if needed
        return det_maps, [n3, n4, n5]

    # ------------------------------------------------------------------
    # Decode (optional utility, depending on your evaluator)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def decode(self, det_maps: List[torch.Tensor], conf_thres: float = 0.25, iou_thres: float = 0.5):
        # Delegate to head's decode if it exists
        if hasattr(self.head, "decode"):
            return self.head.decode(det_maps, conf_thres=conf_thres, iou_thres=iou_thres)
        raise NotImplementedError("Head does not provide decode().")

    # ------------------------------------------------------------------
    # Two-stage keypoint prediction from OBB proposals (used by loss/eval)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def kpt_from_obbs(
        self,
        P: List[torch.Tensor],
        obb_batched: List[torch.Tensor],
        topk: int = 256,
        crop: int = 3,
        min_per_img: int = 2,
        fallback_frac: float = 0.4,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Inputs
        ------
        P:            [P3, P4, P5] neck features (we will use P3 for higher resolution)
        obb_batched:  List over batch; each is (M,5) as (cx, cy, w, h, angle_rad)
        Returns
        -------
        uv_pred: (N, 2) in feature coords (P3 grid). If fallback is used, some KPs may be center-based.
        metas:   list of dicts per sample with keys like {"img_idx", "M": 2x3 affine, "scale": float, ...}
        """
        device = P[0].device
        P3 = P[0]  # (B, C, H, W), stride=8 in typical YOLO11
        B = P3.shape[0]

        # Flatten OBBs into single list with (bix, obb)
        per_img: List[List[torch.Tensor]] = []
        sel_count = 0
        for b in range(B):
            obb = obb_batched[b]
            if obb.numel() == 0:
                per_img.append([])
                continue
            k = max(min_per_img, min(topk, obb.shape[0]))
            per_img.append([obb[:k]])
            sel_count += k

        if sel_count == 0:
            return torch.empty(0, 2, device=device), []

        # Rotated crops over P3
        crops, metas = self.roi(P3, per_img)  # crops: (N, C, S, S), metas: length N
        # TD head -> (u,v) over SxS crop; map to feature coords via metas
        uv_crop = self.kpt_td(crops)  # (N, 2), in [0,S)

        # Convert crop (u,v) to P3 feature coords (x,y)
        uv_all: List[torch.Tensor] = []
        out_metas: List[Dict[str, torch.Tensor]] = []
        for i, meta in enumerate(metas):
            # meta contains: "M": 2x3 (crop->feature) affine, "bix": int, "stride": int (=P3 stride)
            M = meta["M"].to(device=device, dtype=uv_crop.dtype)  # (2,3)
            u, v = uv_crop[i, 0], uv_crop[i, 1]
            # homogeneous [u,v,1]
            xy1 = torch.stack([u, v, torch.ones_like(u, device=device)], dim=0)  # (3,)
            xy_feat = torch.mv(torch.cat([M, torch.tensor([[0,0,1]], device=device, dtype=uv_crop.dtype)], dim=0)[:2], xy1)  # (2,)
            uv_all.append(xy_feat)
            out_metas.append(meta)

        uv_pred = torch.stack(uv_all, dim=0)  # (N,2)
        return uv_pred, out_metas

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def num_params(self) -> int:
        return _num_params(self)

    def schema_signature(self) -> str:
        return _schema_signature(self)
