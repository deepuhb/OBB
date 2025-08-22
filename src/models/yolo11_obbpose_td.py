# src/models/yolo11_obbpose_td.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict

import math
import torch
import torch.nn as nn

# Backbone + blocks (your repo exports conv_bn_act from the backbone file)
from src.models.backbones.cspnext11 import CSPBackbone as CSPBackbone, conv_bn_act
# Neck and head you attached
from src.models.necks.pan_fpn import PANFPN
from src.models.heads.obbpose_head import OBBPoseHead

# Rotated ROI used by TD keypoint head
from src.models.layers.rot_roi import RotatedROIPool

# optional rotated NMS (if torchvision supports it)
try:
    from mmcv.ops import nms_rotated as nms_rotated  # angle in degrees
    HAS_ROT_NMS = True
except Exception:
    HAS_ROT_NMS = False
    nms_rotated = None


# ---------------------- Top-down keypoint head ----------------------
class KptTDHead(nn.Module):
    """
    Small head that predicts (u,v) in [0,1] from rotated crops (C,S,S).
    Output is sigmoid-normalized (u,v).
    """
    def __init__(self, in_ch: int, S: int):
        super().__init__()
        b = 64
        self.net = nn.Sequential(
            conv_bn_act(in_ch, b, 3, 1), nn.MaxPool2d(2),
            conv_bn_act(b, b * 2, 3, 1), nn.MaxPool2d(2),
            conv_bn_act(b * 2, b * 4, 3, 1), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(b * 4, 64), nn.SiLU(inplace=True),
            nn.Linear(64, 2), nn.Sigmoid(),  # -> (u, v) in [0,1]
        )
        self.S = int(S)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns (N, 2) in [0,1]
        return self.net(x)


# ---------------------- Model ----------------------
class YOLO11_OBBPOSE_TD(nn.Module):

    def __init__(
            self,
            num_classes: int,
            width: float = 1.0,
            backbone: nn.Module | None = None,
            neck: nn.Module | None = None,
            head: nn.Module | None = None,
            kpt_crop: int = 3,  # crop size S for TD keypoint head / RotatedROIPool
    ):
        super().__init__()
        # ---- model meta ----
        self.num_classes = int(num_classes)
        self.width = float(width)
        self.kpt_crop = int(kpt_crop)

        # ---- backbone ----
        self.backbone = backbone if backbone is not None else CSPBackbone(width=self.width)

        # Determine backbone output channels (P3,P4,P5). Prefer explicit attribute if available.
        if hasattr(self.backbone, "out_channels"):
            c3, c4, c5 = [int(x) for x in self.backbone.out_channels]  # type: ignore[attr-defined]
        else:
            # Canonical YOLO11/CSPNeXt-11 widths (fallback)
            c3 = int(256 * self.width)
            c4 = int(512 * self.width)
            c5 = int(1024 * self.width)

        # ---- neck (PAN-FPN) ----
        # PANFPN expects ch_in=(c3,c4,c5) and produces three maps with ch_out channels each.
        default_ch_out = int(256 * self.width)
        self.neck: PANFPN = neck if neck is not None else PANFPN(ch_in=(c3, c4, c5), ch_out=default_ch_out)

        # unify channel count leaving the neck
        self.ch_out = int(getattr(self.neck, "ch_out", default_ch_out))

        # ---- head ----
        # OBBPoseHead expects a 3-tuple for per-level channels.
        head_ch = (self.ch_out, self.ch_out, self.ch_out)
        self.head: OBBPoseHead = head if head is not None else OBBPoseHead(
            ch=head_ch,
            num_classes=self.num_classes,
        )

        # ---- top-down keypoint branch (used by kpt_from_obbs) ----
        # Rotated crops from P3 → small conv MLP → (u,v) ∈ [0,1], later scaled to [0,S).
        self.roi = RotatedROIPool(out_size=self.kpt_crop)
        self.kpt_head = KptTDHead(in_ch=self.ch_out, S=self.kpt_crop)

        # (Optional) quick schema print to catch future drift early
        try:
            total = sum(p.numel() for p in self.parameters())
            print(f"[SCHEMA TRAIN] sig=explicit-pan v2 params={total}", flush=True)
            print(f"[SCHEMA TRAIN] backbone_out=({c3},{c4},{c5}) neck_ch_out={self.ch_out}", flush=True)
        except Exception:
            pass

    @torch.no_grad()
    def _feature_pyramid(self, images: torch.Tensor):
        # Convenience split in case you reuse elsewhere
        p3, p4, p5 = self.backbone(images)          # /8, /16, /32
        n3, d4, d5 = self.neck(p3, p4, p5)          # unified channels
        return n3, d4, d5

    def forward(self, images: torch.Tensor):
        n3, d4, d5 = self._feature_pyramid(images)
        det_maps, kpt_maps = self.head([n3, d4, d5])  # keep your original head API
        return det_maps, (n3, d4, d5)

    # ---------------------- TD keypoint crops → (u,v) ----------------------
    @torch.no_grad()
    def kpt_from_obbs(
            self,
            P3: torch.Tensor,
            boxes_list: list[torch.Tensor],
            *,
            scores_list: list[torch.Tensor] | None = None,
            topk: int = 256,
            min_per_img: int = 2,
            fallback_frac: float = 0.4,
    ):
        """
        Select OBBs (optionally score-ranked), crop from P3 with RotatedROIPool,
        and predict (u,v) with the TD keypoint head.

        Args:
            P3:        [B, C, H/8, W/8] feature map from the neck.
            boxes_list: list of length B, each [Ni, 5] (cx, cy, w, h, angle) in image px.
            scores_list: (optional) list of length B, each [Ni] scores to rank boxes.
            topk:      cap on total crops across the batch.
            min_per_img: guarantee at least this many crops per image when possible.
            fallback_frac: if there aren’t enough boxes, take a fraction per image.

        Returns:
            uv_all: (N, 2) in [0, 1]
            metas:  list of dicts for each crop: {"bix": int, "stride": 8, "box": Tensor(5)}
        """
        B = len(boxes_list)
        device = P3.device
        stride = 8  # P3 stride

        # ----- choose indices per image (uses scores if provided; otherwise keep old behavior) -----
        chosen = []
        total_cap = int(topk)
        # first ensure at least min_per_img per image (if available)
        prelim = []
        for bix in range(B):
            boxes = boxes_list[bix]
            if boxes is None or boxes.numel() == 0:
                continue
            n = boxes.shape[0]
            take = min(n, min_per_img)
            if scores_list is not None and bix < len(scores_list) and scores_list[bix] is not None:
                idx = torch.argsort(scores_list[bix].to(device), descending=True)[:take]
            else:
                idx = torch.arange(take, device=device)
            prelim.append([(bix, i.item()) for i in idx])

        prelim = [x for sub in prelim for x in sub]
        chosen.extend(prelim)

        # fill the rest up to topk by global score (if available), else round-robin
        if len(chosen) < total_cap:
            pool = []
            for bix in range(B):
                boxes = boxes_list[bix]
                if boxes is None or boxes.numel() == 0:
                    continue
                n = boxes.shape[0]
                start = min_per_img
                if start >= n:
                    continue
                if scores_list is not None and bix < len(scores_list) and scores_list[bix] is not None:
                    s = scores_list[bix].to(device)
                    idx = torch.argsort(s, descending=True)[start:]
                    pool.extend([(float(s[j].item()), bix, int(j)) for j in idx])
                else:
                    # no scores → FIFO
                    pool.extend([(0.0, bix, j) for j in range(start, n)])

            # sort by score desc if scores exist; FIFO otherwise is already fine
            if any(sc is not None for sc in (scores_list or [])):
                pool.sort(key=lambda t: t[0], reverse=True)

            need = total_cap - len(chosen)
            for _, bix, j in pool[:need]:
                chosen.append((bix, j))

        if not chosen:
            # Nothing to crop
            uv_all = torch.empty(0, 2, device=device)
            metas = []
            return uv_all, metas

        # ----- build ROI boxes in feature coords (divide by stride) -----
        roi_specs = []
        metas = []
        grouped = {}
        for bix, j in chosen:
            grouped.setdefault(bix, []).append(j)

        crops = []
        meta_list = []
        for bix, js in grouped.items():
            boxes_img = boxes_list[bix][js]  # [k,5] (cx,cy,w,h,ang_deg)
            # scale to P3 coordinates
            boxes_feat = boxes_img.clone()
            boxes_feat[:, 0:4] /= stride  # cx,cy,w,h → /8
            # RotatedROIPool expects (batch_idx, cx, cy, w, h, angle_degrees)
            batch_inds = torch.full((boxes_feat.size(0), 1), int(bix), device=device, dtype=boxes_feat.dtype)
            rois = torch.cat([batch_inds, boxes_feat], dim=1)  # [k,6]
            # pool from P3
            pooled = self.roi(P3, rois)  # [k, C, S, S]
            crops.append(pooled)
            for j_local, j_abs in enumerate(js):
                meta_list.append({"bix": int(bix), "stride": stride, "box": boxes_list[bix][j_abs]})

        crops = torch.cat(crops, dim=0)  # (N, C, S, S)
        # ----- predict (u,v) on [0,1] -----
        uv_all = self.kpt_head(crops)  # (N, 2) already sigmoid
        return uv_all, meta_list

    # ---------------------- Export-time decoder ----------------------
    @staticmethod
    def _make_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        yv, xv = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        return xv, yv  # (h, w)

    @torch.no_grad()
    def decode(
        self,
        det_maps: List[torch.Tensor],
        strides: Tuple[int, int, int] = (8, 16, 32),
        conf_thres: float = 0.25,
        iou_thres: float = 0.5,
        max_det: int = 300,
        angle_act: str = "tanh_pi",   # "tanh_pi" -> angle in [-pi/2, pi/2]; "sigmoid_pi" -> [0, pi]
        wh_act: str = "exp",          # "exp" or "softplus"
        use_rotated_nms: Optional[bool] = None,
    ) -> List[torch.Tensor]:
        """
        Decode head outputs into per-image detections with (rotated) NMS.

        Returns
        -------
        per_image: list of [N, 7] tensors:
            [cx, cy, w, h, angle_rad, score, cls]
        """
        if use_rotated_nms is None:
            use_rotated_nms = HAS_ROT_NMS

        B = det_maps[0].shape[0]
        device = det_maps[0].device
        dtype = det_maps[0].dtype
        per_image: List[torch.Tensor] = []

        for b in range(B):
            all_boxes = []
            all_scores = []
            all_labels = []

            for lvl, (pred_l, s) in enumerate(zip(det_maps, strides)):
                # pred_l: (B, C, H, W)
                pl = pred_l[b]  # (C, H, W)
                C, H, W = pl.shape
                assert C >= 7, "det map must have at least 7 channels [tx,ty,tw,th,ta,tobj, tcls0..]"

                tx = pl[0].sigmoid()
                ty = pl[1].sigmoid()

                if wh_act == "exp":
                    tw = pl[2].exp()
                    th = pl[3].exp()
                else:
                    tw = nn.functional.softplus(pl[2])
                    th = nn.functional.softplus(pl[3])

                if angle_act == "tanh_pi":
                    ta = pl[4].tanh() * (0.5 * math.pi)  # radians in [-pi/2, pi/2]
                else:
                    ta = pl[4].sigmoid() * math.pi       # radians in [0, pi]

                tobj = pl[5].sigmoid()                   # (H, W)

                # classes: handle 1 or many
                if C == 7:
                    # single class case: tcls0 is pl[6]
                    tcls = pl[6].sigmoid().unsqueeze(0)  # (1, H, W)
                else:
                    tcls = pl[6:].sigmoid()              # (nc, H, W)

                # grid
                gx, gy = self._make_grid(H, W, device, dtype)
                # decode to pixels
                cx = (gx + tx) * s
                cy = (gy + ty) * s
                w  = tw * s
                h  = th * s

                # scores: obj * class prob (broadcast)
                scores = tobj.unsqueeze(0) * tcls  # (nc, H, W)
                nc = scores.shape[0]

                # filter by conf
                scores_f = scores.flatten(1)  # (nc, H*W)
                keep_mask = scores_f > conf_thres
                if not keep_mask.any():
                    continue

                ys, idx = keep_mask.nonzero(as_tuple=True)  # ys=class, idx=flat index
                sel_scores = scores_f[ys, idx]

                # gather corresponding boxes/angles
                cx_f = cx.flatten()[idx]
                cy_f = cy.flatten()[idx]
                w_f  = w.flatten()[idx]
                h_f  = h.flatten()[idx]
                a_f  = ta.flatten()[idx]  # radians

                # collect
                all_boxes.append(torch.stack([cx_f, cy_f, w_f, h_f, a_f], dim=1))
                all_scores.append(sel_scores)
                all_labels.append(ys)

            if len(all_boxes) == 0:
                per_image.append(torch.zeros((0, 7), device=device, dtype=dtype))
                continue

            boxes = torch.cat(all_boxes, dim=0)    # (N, 5) radians
            scores = torch.cat(all_scores, dim=0)  # (N,)
            labels = torch.cat(all_labels, dim=0)  # (N,)

            # NMS
            if use_rotated_nms and HAS_ROT_NMS:
                # torchvision expects degrees in [-90, 90)
                deg = boxes[:, 4] * (180.0 / math.pi)
                rot_boxes = torch.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], deg], dim=1)
                keep = nms_rotated(rot_boxes, scores, iou_thres)
            else:
                # Fallback: approximate with AABB NMS (coarse)
                try:
                    from torchvision.ops import nms as tv_nms
                    x1 = boxes[:, 0] - boxes[:, 2] * 0.5
                    y1 = boxes[:, 1] - boxes[:, 3] * 0.5
                    x2 = boxes[:, 0] + boxes[:, 2] * 0.5
                    y2 = boxes[:, 1] + boxes[:, 3] * 0.5
                    aabb = torch.stack([x1, y1, x2, y2], dim=1)
                    keep = tv_nms(aabb, scores, iou_thres)
                except Exception:
                    # Last resort: sort by score and just take top-k
                    keep = torch.argsort(scores, descending=True)

            if max_det is not None and keep.numel() > max_det:
                keep = keep[:max_det]

            kept = boxes[keep]
            kept_scores = scores[keep]
            kept_labels = labels[keep].to(kept.dtype)

            out = torch.cat([kept, kept_scores[:, None], kept_labels[:, None]], dim=1)  # (N, 7)
            per_image.append(out)

        return per_image

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        conf_thres: float = 0.25,
        iou_thres: float = 0.5,
        max_det: int = 300,
        strides: Tuple[int, int, int] = (8, 16, 32),
        **decode_kw,
    ) -> List[torch.Tensor]:
        """
        Convenience wrapper: forward → decode.
        """
        det_maps, feats = self.forward(images)
        return self.decode(det_maps, strides=strides, conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det, **decode_kw)
