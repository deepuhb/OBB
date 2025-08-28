# src/models/heads/obbpose_head.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops import nms_rotated as mmcv_nms_rotated


class OBBPoseHead(nn.Module):
    """
    YOLO-style DFL-only OBB head (+tiny kpt head kept for compatibility).

    Per-level detection layout (C = 2 + 2*(reg_max+1) + 1 + 1 + num_classes):
        [ tx, ty,
          dfl_w[0..reg_max], dfl_h[0..reg_max],
          ang_logit, obj, (cls0..cls{num_classes-1}) ]

    - tx, ty: raw offsets in [0,1) after sigmoid in decoder
    - dfl_w/h: logits over 0..reg_max (expected value -> width/height in *stride units*)
    - ang_logit: single logit; decoder maps to angle ∈ [-π/2, π/2)
    - obj, cls: raw logits (BCE-with-logits in loss; sigmoid at decode)
    """

    def __init__(
        self,
        ch: Tuple[int, int, int],
        strides: Tuple[int, int, int],
        num_classes: int,
        reg_max: int = 16,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.nc = int(num_classes)
        self.reg_max = int(reg_max)
        self.nbins = self.reg_max + 1
        self.strides = tuple(int(s) for s in strides)  # (8,16,32)

        # cached DFL bins (moves with device/dtype; no optimizer state)
        self.register_buffer(
            "dfl_bins",
            torch.arange(self.nbins, dtype=torch.float32).view(1, self.nbins, 1, 1),
            persistent=False,
        )

        c3, c4, c5 = [int(x) for x in ch]
        det_out = 2 + 2 * self.nbins + 1 + 1 + self.num_classes  # tx,ty + DFL(w,h) + ang + obj + cls
        kpt_out = 3  # (kpx, kpy, score) — not used by decoder, kept to maintain model interface

        # Simple 3x conv heads
        def det_head(cin):  # detection head
            return nn.Sequential(
                nn.Conv2d(cin, cin, 3, 1, 1), nn.BatchNorm2d(cin), nn.SiLU(inplace=True),
                nn.Conv2d(cin, det_out, 1, 1, 0)
            )

        def kpt_head(cin):  # tiny kpt head
            return nn.Sequential(
                nn.Conv2d(cin, cin, 3, 1, 1), nn.BatchNorm2d(cin), nn.SiLU(inplace=True),
                nn.Conv2d(cin, kpt_out, 1, 1, 0)
            )

        self.det3, self.det4, self.det5 = det_head(c3), det_head(c4), det_head(c5)
        self.kp3,  self.kp4,  self.kp5  = kpt_head(c3), kpt_head(c4), kpt_head(c5)

        self._assert_once = False
        self._printed_ang_norm = False

    # ---------- helpers ----------
    @staticmethod
    def _le90(w: torch.Tensor, h: torch.Tensor, ang: torch.Tensor):
        """Enforce w ≥ h and wrap angle to [-π/2, π/2)."""
        swap = w < h
        w2 = torch.where(swap, h, w)
        h2 = torch.where(swap, w, h)
        ang2 = torch.where(swap, ang + math.pi / 2.0, ang)
        ang2 = torch.remainder(ang2 + math.pi / 2.0, math.pi) - math.pi / 2.0
        return w2, h2, ang2

    def _split_det_map(self, p: torch.Tensor):
        # p: (B, C, H, W) ; channels = [tx,ty, dflw(R), dflh(R), ang, obj, (cls...)]
        B, C, H, W = p.shape
        nb = self.reg_max + 1
        idx = 0
        tx = p[:, idx:idx + 1]
        idx += 1
        ty = p[:, idx:idx + 1]
        idx += 1
        dflw = p[:, idx:idx + nb]
        idx += nb
        dflh = p[:, idx:idx + nb]
        idx += nb
        ang = p[:, idx:idx + 1]
        idx += 1
        obj = p[:, idx:idx + 1]
        idx += 1
        cls = p[:, idx:idx + self.num_classes] if self.num_classes > 1 else None
        return {"tx": tx, "ty": ty, "dflw": dflw, "dflh": dflh, "ang": ang, "obj": obj, "cls": cls}

    @staticmethod
    def rotated_nms_mmcv(boxes_xywht_rad: torch.Tensor,
                         scores: torch.Tensor,
                         iou: float,
                         max_det: int,
                         conf_thres: float,
                         clockwise: bool = False):
        """
        boxes_xywht_rad: (N,5) [xc,yc,w,h,theta_rad], float32, on same device as scores
        scores:          (N,)   float32
        iou:             float in [0,1]
        max_det:         int
        conf_thres:      float in [0,1]
        clockwise:       MMCV expects True if +angle is clockwise; set False for the usual CCW.
        """
        if boxes_xywht_rad.numel() == 0:
            return boxes_xywht_rad, scores, torch.empty(0, dtype=torch.long, device=boxes_xywht_rad.device)

        # 1) confidence filter
        mask = scores >= conf_thres
        if not mask.any():
            return (boxes_xywht_rad.new_zeros((0, 5)),
                    scores.new_zeros((0,)),
                    torch.empty(0, dtype=torch.long, device=boxes_xywht_rad.device))

        boxes = boxes_xywht_rad[mask].float().contiguous()
        sc = scores[mask].float().contiguous()

        # 2) (optional) pre-NMS top-k for speed
        if boxes.shape[0] > max_det * 5:
            topk = min(boxes.shape[0], max_det * 5)
            tk_scores, tk_idx = torch.topk(sc, k=topk, dim=0, largest=True, sorted=True)
            boxes, sc = boxes[tk_idx], tk_scores

        # 3) MMCV rotated NMS; NOTE: angle must be radians; set clockwise flag correctly
        kept_dets, keep_idx = mmcv_nms_rotated(boxes, sc, float(iou), labels=None, clockwise=clockwise)
        keep_idx = keep_idx.to(torch.long)  # ensure usable as indices

        # kept_dets is (M, 6): [xc, yc, w, h, theta_rad, score]
        # We want to return boxes/scores aligned with the original filtered set (after mask/topk)
        # Limit to max_det
        if kept_dets.size(0) > max_det:
            kept_dets = kept_dets[:max_det]
            keep_idx = keep_idx[:max_det]

        out_boxes = kept_dets[:, :5]
        out_scores = kept_dets[:, 5]

        return out_boxes, out_scores, keep_idx

    @staticmethod
    def _nms_aabb_fallback(boxes_xywht_rad: torch.Tensor,
                           scores: torch.Tensor,
                           iou_thres: float = 0.50,
                           max_det: int = 300) -> torch.Tensor:
        """
        Very simple axis-aligned NMS fallback: convert rotated boxes to AABB and run NMS.
        Returns keep indices into the input boxes.
        """
        if boxes_xywht_rad.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=boxes_xywht_rad.device)

        # convert (xc, yc, w, h, theta) to axis-aligned [x1,y1,x2,y2] that encloses the rotated rect
        xc, yc, w, h, _ = boxes_xywht_rad.unbind(dim=1)
        x1 = xc - w * 0.5
        y1 = yc - h * 0.5
        x2 = xc + w * 0.5
        y2 = yc + h * 0.5
        aabb = torch.stack([x1, y1, x2, y2], dim=1)

        try:
            from torchvision.ops import nms
            keep = nms(aabb, scores, float(iou_thres))
        except Exception:
            # naive NMS if torchvision isn't available
            order = scores.sort(descending=True).indices
            keep_list = []
            while order.numel() > 0 and len(keep_list) < max_det:
                i = order[0]
                keep_list.append(i.item())
                if order.numel() == 1:
                    break
                rest = order[1:]
                xx1 = torch.maximum(aabb[rest, 0], aabb[i, 0])
                yy1 = torch.maximum(aabb[rest, 1], aabb[i, 1])
                xx2 = torch.minimum(aabb[rest, 2], aabb[i, 2])
                yy2 = torch.minimum(aabb[rest, 3], aabb[i, 3])
                inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
                area_i = (aabb[i, 2] - aabb[i, 0]).clamp(min=0) * (aabb[i, 3] - aabb[i, 1]).clamp(min=0)
                area_r = (aabb[rest, 2] - aabb[rest, 0]).clamp(min=0) * (aabb[rest, 3] - aabb[rest, 1]).clamp(min=0)
                iou = inter / (area_i + area_r - inter + 1e-9)
                order = rest[iou <= float(iou_thres)]
            keep = torch.tensor(keep_list, device=boxes_xywht_rad.device, dtype=torch.long)

        return keep[:max_det].to(torch.long)

    @staticmethod
    def _ev_from_dfl(logits: torch.Tensor) -> torch.Tensor:
        # logits: (B, nbins, H, W) -> (B,1,H,W) in bin units
        probs = torch.softmax(logits, dim=1)
        bins = torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype).view(1, -1, 1, 1)
        return (probs * bins).sum(dim=1, keepdim=True)

    def _log_bounds(self, level: int, stride: float):
        # use per-level ranges if provided; else safe defaults
        if hasattr(self, "dfl_log_minmax") and self.dfl_log_minmax:
            vmin, vmax = self.dfl_log_minmax[level]
            return float(vmin), float(vmax)
        # default: cover about [2px..64px] per level (prevents collapse to image-scale)
        return math.log(2.0 / stride), math.log(64.0 / stride)



    # ---------- forward ----------
    def forward(self, feats: List[torch.Tensor]):
        """
        Args:
            feats: [P3, P4, P5]
        Returns:
            det_maps: list of (B, 2+2*(R+1)+1+1+nc, H, W)
            kpt_maps: list of (B, 3, H, W)   (kept for compatibility with your model/loss)
        """
        assert isinstance(feats, (list, tuple)) and len(feats) == 3, \
            "OBBPoseHead.forward expects 3 feature maps [P3,P4,P5]"
        p3, p4, p5 = feats
        det_maps = [self.det3(p3), self.det4(p4), self.det5(p5)]
        kpt_maps = [self.kp3(p3),  self.kp4(p4),  self.kp5(p5)]

        if not self._assert_once:
            expected = 2 + 2 * self.nbins + 1 + 1 + self.num_classes
            for i, lvl in enumerate(("P3", "P4", "P5")):
                C = int(det_maps[i].shape[1])
                assert C == expected, (
                    f"OBBPoseHead({lvl}) channels={C}, expected {expected}="
                    f"[tx,ty,dflw({self.nbins}),dflh({self.nbins}),ang,obj,(cls...)]"
                )
            self._assert_once = True

        if not self._printed_ang_norm:
            is_main = True
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                    is_main = False
            except Exception:
                pass
            if is_main:
                print("[ASSERT] angle uses single-logit (YOLO-style).")
            self._printed_ang_norm = True

        return det_maps, kpt_maps

    # ---------- decoder (DFL-only) ----------
    @torch.no_grad()
    @torch.no_grad()
    def decode_obb_from_pyramids(
            self,
            det_maps: list,
            imgs: torch.Tensor,
            *,
            strides=None,
            conf_thres: float = 0.25,  # synonyms accepted via **kwargs
            iou: float = 0.50,
            use_nms: bool = True,
            max_det: int = 300,
            **kwargs,
    ):
        """
        Decode YOLO-style OBB predictions from pyramid maps (DFL-only).
        Returns a list of length B, each with dict{boxes:(N,5)[xc,yc,w,h,theta_rad], scores:(N,), labels:(N,)}

        - Uses self._split_det_map() layout: [tx,ty, dflw, dflh, ang, obj, (cls...)]
        - DFL width/height are turned into pixels with the same per-level log mapping used in loss (_log_bounds).
        - Angle is a single logit -> tanh * (π/2), consistent with training.
        - NMS is class-agnostic by default (matches your current setup).
        """
        # --- normalize common kwarg aliases (prevents "unexpected keyword" errors) ---
        if 'conf_thresh' in kwargs:  # alias
            conf_thres = float(kwargs['conf_thresh'])
        if 'score_thr' in kwargs:
            conf_thres = float(kwargs['score_thr'])
        if 'score_threshold' in kwargs:
            conf_thres = float(kwargs['score_threshold'])
        if 'nms_iou' in kwargs:
            iou = float(kwargs['nms_iou'])
        if 'iou_thresh' in kwargs:
            iou = float(kwargs['iou_thresh'])
        if 'max_det_per_image' in kwargs:
            max_det = int(kwargs['max_det_per_image'])
        if strides is None:
            # prefer instance strides when not provided by caller
            strides = getattr(self, 'strides', (8, 16, 32))

        device = det_maps[0].device
        B, _, Himg, Wimg = imgs.shape
        reg_bins = self.reg_max + 1
        out_per_img = [[] for _ in range(B)]

        # small grid cache per level
        def make_grid(h, w, dev, dt):
            gy, gx = torch.meshgrid(
                torch.arange(h, device=dev, dtype=dt),
                torch.arange(w, device=dev, dtype=dt),
                indexing="ij"
            )
            return gx.view(1, 1, h, w), gy.view(1, 1, h, w)

        for li, (pyr, s) in enumerate(zip(det_maps, strides)):
            p = pyr.contiguous()  # (B, C, H, W)
            b, c, h, w = p.shape
            mp = self._split_det_map(p)  # dict of tensors
            tx, ty, dflw, dflh = mp["tx"], mp["ty"], mp["dflw"], mp["dflh"]
            ang_logit, obj, cls = mp["ang"], mp["obj"], mp["cls"]

            # centers (in pixels)
            gx, gy = make_grid(h, w, p.device, p.dtype)
            cx = (gx + tx.sigmoid()) * float(s)
            cy = (gy + ty.sigmoid()) * float(s)

            # DFL expected value -> (log size in stride units) -> pixels
            vmin, vmax = self._log_bounds(li, float(s))
            # softmax over DFL bins then expected value
            dflw_prob = torch.softmax(dflw, dim=1)
            dflh_prob = torch.softmax(dflh, dim=1)
            bins = torch.arange(reg_bins, device=p.device, dtype=p.dtype).view(1, reg_bins, 1, 1)
            w_ev = (dflw_prob * bins).sum(dim=1, keepdim=True)  # (B,1,H,W)
            h_ev = (dflh_prob * bins).sum(dim=1, keepdim=True)

            w_px = torch.exp(vmin + (w_ev / max(reg_bins - 1, 1)) * (vmax - vmin)) * float(s)
            h_px = torch.exp(vmin + (h_ev / max(reg_bins - 1, 1)) * (vmax - vmin)) * float(s)

            # angle, score
            ang = ang_logit.tanh() * (math.pi / 2.0)
            obj_p = obj.sigmoid()
            if self.num_classes > 1 and cls is not None:
                cls_p = cls.sigmoid()  # (B,nc,H,W)
                cls_max, cls_id = cls_p.max(dim=1)  # (B,H,W), (B,H,W)
                score = obj_p[:, 0] * cls_max  # (B,H,W)
            else:
                score = obj_p[:, 0]  # (B,H,W)
                cls_id = torch.zeros((B, h, w), device=device, dtype=torch.long)

            keep = (score >= float(conf_thres))  # (B,H,W) mask

            for bi in range(B):
                kb = keep[bi]
                if not kb.any():
                    out_per_img[bi].append({
                        "boxes": torch.zeros((0, 5), device=device),
                        "scores": torch.zeros((0,), device=device),
                        "labels": torch.zeros((0,), device=device, dtype=torch.long),
                    })
                    continue

                x = cx[bi, 0][kb]
                y = cy[bi, 0][kb]
                ww = w_px[bi, 0][kb]
                hh = h_px[bi, 0][kb]
                aa = ang[bi, 0][kb]
                sc = score[bi][kb]
                lb = cls_id[bi][kb] if (self.num_classes > 1) else torch.zeros_like(sc, dtype=torch.long)

                boxes = torch.stack([x, y, ww, hh, aa], dim=1)  # radians
                if use_nms and boxes.numel() > 0:
                    try:
                        # mmcv nms_rotated expects angles in DEGREES
                        boxes_deg = boxes.clone()
                        boxes_deg[:, 4] = boxes_deg[:, 4] * (180.0 / math.pi)
                        # returns (kept_dets[N,6], keep_idx[N]); we only need indices
                        kept_dets, keep_idx = mmcv_nms_rotated(boxes_deg, sc, float(iou), clockwise=False)
                        keep_idx = keep_idx.to(torch.long)[:max_det]
                    except Exception:
                        keep_idx = self._nms_aabb_fallback(boxes, sc, iou_thres=float(iou), max_det=max_det)
                    boxes, sc, lb = boxes[keep_idx], sc[keep_idx], lb[keep_idx]

                out_per_img[bi].append({"boxes": boxes, "scores": sc, "labels": lb})

        # --- concat levels, then (optional) global NMS per image ---
        final = []
        for bi in range(B):
            if not out_per_img[bi]:
                final.append({"boxes": torch.zeros((0, 5), device=device),
                              "scores": torch.zeros((0,), device=device),
                              "labels": torch.zeros((0,), device=device, dtype=torch.long)})
                continue
            boxes = torch.cat([o["boxes"] for o in out_per_img[bi]], dim=0)
            score = torch.cat([o["scores"] for o in out_per_img[bi]], dim=0)
            label = torch.cat([o["labels"] for o in out_per_img[bi]], dim=0)

            if use_nms and boxes.numel():
                try:
                    boxes_deg = boxes.clone()
                    boxes_deg[:, 4] = boxes_deg[:, 4] * (180.0 / math.pi)
                    kept_dets, keep_idx = mmcv_nms_rotated(boxes_deg, score, float(iou), clockwise=False)
                    keep_idx = keep_idx.to(torch.long)[:max_det]
                except Exception:
                    keep_idx = self._nms_aabb_fallback(boxes, score, iou_thres=float(iou), max_det=max_det)
                boxes, score, label = boxes[keep_idx], score[keep_idx], label[keep_idx]

            final.append({"boxes": boxes, "scores": score, "labels": label})
        return final