# src/models/heads/obbpose_head.py
from __future__ import annotations
from typing import List, Tuple, Optional
import math
import torch
import torch.nn as nn
from torch import amp


class OBBPoseHead(nn.Module):
    """
    YOLO-style OBB + tiny KPT head.

    Detection layout per level (C = 2 + 2*(reg_max+1) + 1 + 1 + nc):
        [ tx, ty,
          dfl_w[0..reg_max], dfl_h[0..reg_max],
          ang_logit, obj, (cls0..cls{nc-1}) ]

    - tx, ty: raw offsets (sigmoid in decoder)
    - dfl_w/h: logits over 0..reg_max (expected value -> width/height in stride units)
    - ang_logit: single logit; decoder maps to [-pi/2, pi/2) then LE-90 canonicalisation
    - obj, cls: raw logits
    """

    def __init__(self, ch: Tuple[int, int, int],strides, num_classes: int, reg_max: int = 16) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.nc = int(num_classes)
        self.reg_max = int(reg_max)
        self.nbins = self.reg_max + 1
        self.strides = tuple(int(s) for s in strides)  # e.g. (8, 16, 32)

        # bins 0..reg_max, cached as a non-persistent buffer (auto-moves with module/device)
        self.register_buffer("dfl_bins", torch.arange(self.nbins, dtype=torch.float32).view(1, self.nbins, 1, 1),
                             persistent=False)

        c3, c4, c5 = [int(x) for x in ch]
        det_out = 2 + 2 * self.nbins + 1 + 1 + self.num_classes  # tx,ty + DFL(w,h) + ang + obj + cls
        kpt_out = 3  # (kpx, kpy, kpscore) tiny head kept for compatibility even if unused

        # P3/P4/P5 detection heads
        self.det3 = nn.Sequential(
            nn.Conv2d(c3, c3, 3, 1, 1), nn.BatchNorm2d(c3), nn.SiLU(inplace=True),
            nn.Conv2d(c3, det_out, 1, 1, 0)
        )
        self.det4 = nn.Sequential(
            nn.Conv2d(c4, c4, 3, 1, 1), nn.BatchNorm2d(c4), nn.SiLU(inplace=True),
            nn.Conv2d(c4, det_out, 1, 1, 0)
        )
        self.det5 = nn.Sequential(
            nn.Conv2d(c5, c5, 3, 1, 1), nn.BatchNorm2d(c5), nn.SiLU(inplace=True),
            nn.Conv2d(c5, det_out, 1, 1, 0)
        )

        # Tiny keypoint heads (safe even if unused)
        self.kp3 = nn.Sequential(
            nn.Conv2d(c3, c3, 3, 1, 1), nn.BatchNorm2d(c3), nn.SiLU(inplace=True),
            nn.Conv2d(c3, kpt_out, 1, 1, 0)
        )
        self.kp4 = nn.Sequential(
            nn.Conv2d(c4, c4, 3, 1, 1), nn.BatchNorm2d(c4), nn.SiLU(inplace=True),
            nn.Conv2d(c4, kpt_out, 1, 1, 0)
        )
        self.kp5 = nn.Sequential(
            nn.Conv2d(c5, c5, 3, 1, 1), nn.BatchNorm2d(c5), nn.SiLU(inplace=True),
            nn.Conv2d(c5, kpt_out, 1, 1, 0)
        )

        # --- lightweight buffers & caches (save allocs every decode) ---
        self._grid_cache = {}  # (device, H, W) -> (gx, gy) cached 1x1xHxW float32

        # one-time debug guards
        self._assert_once = False
        self._printed_ang_norm = False
        self._dbg_decode_once = False

        self._init_det_bias(self.det3)
        self._init_det_bias(self.det4)
        self._init_det_bias(self.det5)

        self.dfl_log_minmax = (
            (math.log(1.0/8),  math.log(128.0/8)),   # P3
            (math.log(1.0/16), math.log(128.0/16)),  # P4
            (math.log(1.0/32), math.log(128.0/32)),  # P5
        )

    def _init_det_bias(self, mod: nn.Sequential) -> None:
        """Init objectness bias ~1% prior; keep other heads neutral."""
        final = mod[-1]
        assert isinstance(final, nn.Conv2d)
        with torch.no_grad():
            b = final.bias
            # make sure bias exists and is the right size
            if b is None or b.numel() == 0:
                final.bias = nn.Parameter(torch.zeros_like(final.weight[:, 0, 0, 0]))
                b = final.bias
            b.zero_()
            # channels: [tx,ty, dflw(R+1), dflh(R+1), ang, obj, (cls...)]
            obj_idx = 2 + 2 * (self.reg_max + 1) + 1  # after ang
            if b.numel() > obj_idx:
                b[obj_idx] = -3.0  # ~logit(1%) == -4.6, use -3 for a bit less conservative start

    # ---------- helpers ----------
    def _split_det_map(self, x: torch.Tensor):
        """
        Split a det tensor (B,C,H,W) into parts according to the new layout.
        Returns dict with keys: 'tx','ty','dflw','dflh','ang','obj','cls'
        """
        B, C, H, W = x.shape
        s0 = 0
        tx = x[:, s0:s0 + 1]; s0 += 1
        ty = x[:, s0:s0 + 1]; s0 += 1
        dflw = x[:, s0:s0 + self.nbins]; s0 += self.nbins
        dflh = x[:, s0:s0 + self.nbins]; s0 += self.nbins
        ang = x[:, s0:s0 + 1]; s0 += 1
        obj = x[:, s0:s0 + 1]; s0 += 1
        cls = x[:, s0:] if (C - s0) > 0 else None
        return {"tx": tx, "ty": ty, "dflw": dflw, "dflh": dflh, "ang": ang, "obj": obj, "cls": cls}

    def _get_grid(self, h: int, w: int, device: torch.device, dtype: torch.dtype):
        """
        Returns cached (gx, gy) shaped (1,1,H,W) without expanding to B.
        Stored in float32 to keep softmax/EV stable; cast on use if needed.
        """
        key = (device, h, w)
        gx, gy = self._grid_cache.get(key, (None, None))
        if gx is None or gy is None:
            gy_, gx_ = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij",
            )
            gx = gx_.view(1, 1, h, w)
            gy = gy_.view(1, 1, h, w)
            self._grid_cache[key] = (gx, gy)
        # cast view to requested dtype on the fly (no extra storage)
        return gx.to(dtype), gy.to(dtype)

    @staticmethod
    def _le90(w: torch.Tensor, h: torch.Tensor, ang: torch.Tensor):
        """Ensure w >= h; rotate θ by +π/2 when swapped; wrap θ to [-π/2, π/2)."""
        swap = w < h
        w2 = torch.where(swap, h, w)
        h2 = torch.where(swap, w, h)
        ang2 = torch.where(swap, ang + math.pi / 2.0, ang)
        ang2 = torch.remainder(ang2 + math.pi / 2.0, math.pi) - math.pi / 2.0
        return w2, h2, ang2

    def _ev_from_dfl(self, logits: torch.Tensor):
        """
        Expected value from DFL logits along channel dim (nbins).
        logits: (B, nbins, H, W)  -> returns: (B,1,H,W) in bin units.
        Uses a cached bin buffer to avoid per-call allocations.
        """
        # softmax in native dtype
        probs = torch.softmax(logits, dim=1)
        # match cached bins to current dtype/device without reallocating shape
        bins = self.dfl_bins.to(device=logits.device, dtype=logits.dtype, non_blocking=True)
        return (probs * bins).sum(dim=1, keepdim=True)

    def _safe_exp(self, x: torch.Tensor, max_val: float = 15.0) -> torch.Tensor:
        # exp(log-wh) guard to avoid infs; clamp pre-exp in log-space
        return torch.exp(x.clamp(min=-max_val, max=max_val))

    @staticmethod
    def _make_grid(h: int, w: int, device: torch.device, dtype: torch.dtype):
        gy, gx = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing='ij'
        )
        # (1,1,h,w) for easy broadcasting with (B,1,h,w)
        return gx.unsqueeze(0).unsqueeze(0), gy.unsqueeze(0).unsqueeze(0)

    def _split_map(self, dm: torch.Tensor) -> dict:
        """Split a single level map into named tensors (DFL-only)."""
        B, C, H, W = dm.shape
        nb = self.reg_max + 1
        nc = int(getattr(self, "nc", 1))
        i = 0
        out = {}
        out["tx"] = dm[:, i, ...]
        i += 1  # (B,H,W)
        out["ty"] = dm[:, i, ...]
        i += 1
        out["dflw"] = dm[:, i:i + nb, ...]
        i += nb  # (B,nb,H,W)
        out["dflh"] = dm[:, i:i + nb, ...]
        i += nb
        out["ang"] = dm[:, i, ...]
        i += 1
        out["obj"] = dm[:, i:i + 1, ...]
        i += 1  # (B,1,H,W)
        if nc > 1 and (i + nc) <= C:
            out["cls"] = dm[:, i:i + nc, ...]  # (B,nc,H,W)
        return out

    # ---------- forward ----------
    def forward(self, feats: List[torch.Tensor]):
        """
        Args:
            feats: [P3, P4, P5]
        Returns:
            det_maps: list of (B, 2+2*(R+1)+1+1+nc, H, W)
            kpt_maps: list of (B, 3, H, W)
        """
        assert isinstance(feats, (list, tuple)) and len(feats) == 3, \
            "OBBPoseHead.forward expects 3 feature maps [P3,P4,P5]"
        p3, p4, p5 = feats
        det_maps = [self.det3(p3), self.det4(p4), self.det5(p5)]
        kpt_maps = [self.kp3(p3), self.kp4(p4), self.kp5(p5)]

        # one-time channel assertion
        if not self._assert_once:
            expected = 2 + 2 * self.nbins + 1 + 1 + self.num_classes
            for i, lvl in enumerate(("P3", "P4", "P5")):
                C = int(det_maps[i].shape[1])
                assert C == expected, (
                    f"OBBPoseHead({lvl}) channels={C}, expected {expected}="
                    f"[tx,ty,dflw({self.nbins}),dflh({self.nbins}),ang,obj,(cls...)]"
                )
            self._assert_once = True

        # single-logit angle notice (print once on rank-0)
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

    def _nms_rotated(self, boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
        """
        boxes: (N,5) in (cx, cy, w, h, angle_rad) [LE-90]
        scores: (N,)
        returns keep indices (1D LongTensor)

        Primary: mmcv.ops.nms_rotated
        Fallbacks: torchvision.ops.nms (AABB) -> pure-Py AABB NMS
        """
        if boxes.numel() == 0:
            return boxes.new_zeros((0,), dtype=torch.long)

        # Convert angles to degrees for mmcv
        rb = boxes.detach().clone()
        rb[:, 4] = rb[:, 4] * (180.0 / math.pi)  # rad -> deg

        # Ensure dtype/device friendly for MMCV
        rb = rb.contiguous()
        scores = scores.contiguous()

        # 1) MMCV rotated NMS
        try:
            from mmcv.ops import nms_rotated as mmcv_nms_rotated
            out = mmcv_nms_rotated(rb, scores, float(iou_thres))
            # mmcv can return either (dets, keep) or just keep depending on version
            if isinstance(out, tuple):
                # dets: (M, 6) [xc, yc, w, h, angle_deg, score], keep: LongTensor idx into input
                _, keep = out
            else:
                keep = out
            return keep.to(dtype=torch.long)
        except Exception:
            pass

        # 2) AABB NMS via torchvision (coarse fallback)
        try:
            from torchvision.ops import nms
            x1 = boxes[:, 0] - boxes[:, 2] * 0.5
            y1 = boxes[:, 1] - boxes[:, 3] * 0.5
            x2 = boxes[:, 0] + boxes[:, 2] * 0.5
            y2 = boxes[:, 1] + boxes[:, 3] * 0.5
            abox = torch.stack([x1, y1, x2, y2], dim=1)
            keep = nms(abox, scores, float(iou_thres))
            return keep
        except Exception:
            pass

        # 3) Pure-Python AABB NMS (last resort, slower)
        x1 = boxes[:, 0] - boxes[:, 2] * 0.5
        y1 = boxes[:, 1] - boxes[:, 3] * 0.5
        x2 = boxes[:, 0] + boxes[:, 2] * 0.5
        y2 = boxes[:, 1] + boxes[:, 3] * 0.5
        order = scores.sort(descending=True).indices
        keep_list = []
        while order.numel() > 0:
            i = int(order[0])
            keep_list.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])
            w = (xx2 - xx1).clamp_(min=0)
            h = (yy2 - yy1).clamp_(min=0)
            inter = w * h
            area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
            area_r = (x2[rest] - x1[rest]) * (y2[rest] - y1[rest])
            iou = inter / (area_i + area_r - inter + 1e-7)
            order = rest[iou <= float(iou_thres)]
        return torch.as_tensor(keep_list, device=boxes.device, dtype=torch.long)

    # ---------- decoder ----------
    @torch.no_grad()
    def decode_obb_from_pyramids(
            self,
            det_maps: list[torch.Tensor],
            imgs: torch.Tensor,
            conf_thres: float = 0.01,  # keep low to avoid "empty preds"
            iou_thres: float = 0.50,
            max_det: int = 300,
            use_nms: bool = True,
            **kwargs,  # tolerate alias kwargs from various evaluators
    ):
        # --- tolerate legacy names ---
        if "score_thr" in kwargs and "conf_thres" not in locals():
            conf_thres = float(kwargs["score_thr"])
        if "iou_thr" in kwargs and "iou_thres" not in locals():
            iou_thres = float(kwargs["iou_thr"])
        if "max_det_per_img" in kwargs and "max_det" not in locals():
            max_det = int(kwargs["max_det_per_img"])

        device = det_maps[0].device
        B = int(imgs.shape[0]) if imgs is not None else int(det_maps[0].shape[0])
        strides = tuple(int(s) for s in getattr(self, "strides", (8, 16, 32)))
        reg_max = int(getattr(self, "reg_max", 16))
        nb = reg_max + 1
        nc = int(getattr(self, "nc", 1))

        # per-level log bins (vector) used to turn DFL logits -> expected log-size
        log_bins = []
        for li, dm in enumerate(det_maps):
            H, W = int(dm.shape[-2]), int(dm.shape[-1])
            if hasattr(self, "dfl_log_minmax"):
                lmin, lmax = self.dfl_log_minmax[li]
            else:
                # safe default
                s = float(strides[li])
                lmin, lmax = math.log(1.0 / s), math.log(128.0 / s)
            idx = torch.arange(nb, device=device, dtype=torch.float32)
            log_bins.append(lmin + (lmax - lmin) * (idx / float(reg_max)))

        all_boxes = [[] for _ in range(B)]
        all_scores = [[] for _ in range(B)]
        all_labels = [[] for _ in range(B)]

        for li, (dm, s, bins) in enumerate(zip(det_maps, strides, log_bins)):
            mp = self._split_map(dm)  # DFL-only layout
            B, _, H, W = dm.shape

            # grid
            yy, xx = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij",
            )
            xx = xx.reshape(1, 1, H, W).float()
            yy = yy.reshape(1, 1, H, W).float()

            # centers
            sx = mp["tx"].sigmoid()  # (B,H,W)
            sy = mp["ty"].sigmoid()
            cx = (xx + sx.unsqueeze(1)) * float(s)  # (B,1,H,W) broadcast
            cy = (yy + sy.unsqueeze(1)) * float(s)

            # sizes (DFL -> expected log size -> exp -> pixels)
            pw = (mp["dflw"].softmax(dim=1) * bins.view(1, nb, 1, 1)).sum(dim=1)  # (B,H,W)
            ph = (mp["dflh"].softmax(dim=1) * bins.view(1, nb, 1, 1)).sum(dim=1)
            w = pw.exp() * float(s)
            h = ph.exp() * float(s)

            # angle single-logit -> [-pi/2, pi/2)
            ang = (mp["ang"].sigmoid() * math.pi) - (math.pi / 2.0)  # (B,H,W)

            # canonicalise LE-90: ensure w>=h and wrap angle
            swap = (w < h)
            w2 = torch.where(swap, h, w)
            h2 = torch.where(swap, w, h)
            ang2 = torch.where(swap, ang + (math.pi / 2.0), ang)
            ang2 = (ang2 + math.pi / 2.0) % math.pi - math.pi / 2.0

            # scores
            obj = mp["obj"].sigmoid().squeeze(1)  # (B,H,W)
            if "cls" in mp and nc > 1:
                cls = mp["cls"].sigmoid().amax(dim=1)  # (B,H,W)
                score = obj * cls
                if hasattr(mp, "cls"):  # silence linters
                    pass
            else:
                score = obj

            # filter by conf
            keep = score > float(conf_thres)
            if not keep.any():
                continue

            # collect per-image
            for b in range(B):
                kb = keep[b]  # (H,W)
                if not kb.any():
                    continue
                cx_b = cx[b, 0][kb]
                cy_b = cy[b, 0][kb]
                w_b = w2[b][kb]
                h_b = h2[b][kb]
                a_b = ang2[b][kb]
                s_b = score[b][kb]

                # labels: argmax if multi-class, else zeros
                if "cls" in mp and nc > 1:
                    lab_b = mp["cls"][b, :, :, :].sigmoid().permute(1, 2, 0)  # (H,W,nc)
                    lab_b = lab_b[kb].argmax(dim=1).to(torch.long)
                else:
                    lab_b = torch.zeros_like(s_b, dtype=torch.long)

                # mmcv.nms_rotated expects angle in degrees
                boxes_b = torch.stack([cx_b, cy_b, w_b, h_b, torch.rad2deg(a_b)], dim=1)  # (N,5)
                all_boxes[b].append(boxes_b)
                all_scores[b].append(s_b)
                all_labels[b].append(lab_b)

        # concatenate and NMS per image
        out = []
        for b in range(B):
            if not all_boxes[b]:
                out.append({
                    "boxes": torch.zeros((0, 5), device=device),
                    "scores": torch.zeros((0,), device=device),
                    "labels": torch.zeros((0,), device=device, dtype=torch.long),
                })
                continue

            boxes = torch.cat(all_boxes[b], dim=0)
            scores = torch.cat(all_scores[b], dim=0)
            labels = torch.cat(all_labels[b], dim=0)

            # NMS
            if use_nms and boxes.numel():
                try:
                    from mmcv.ops import nms_rotated
                    # nms_rotated takes (boxes, scores, iou_thr) with boxes=(x,y,w,h,deg)
                    keep, _ = nms_rotated(boxes, scores, float(iou_thres))
                    keep = keep[:max_det]
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                except Exception:
                    # fallback: simple top-k (keeps duplicates but unblocks training)
                    k = min(int(max_det), int(scores.numel()))
                    topk = torch.topk(scores, k=k, largest=True, sorted=False).indices
                    boxes, scores, labels = boxes[topk], scores[topk], labels[topk]

            out.append({"boxes": boxes, "scores": scores, "labels": labels})
        return out