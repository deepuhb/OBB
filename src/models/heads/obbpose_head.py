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
        """
        Split a detection map with channels [tx,ty,tw,th,sin,cos,obj,(cls...)]
        into a dict of named tensors. Keeps compatibility with your loss.
        """
        B, C, H, W = dm.shape
        expect = 7 + self.nc
        assert C == expect, f"det map channels = {C}, expected {expect} (7+nc)"
        i = 0
        tx = dm[:, i:i + 1]
        i += 1
        ty = dm[:, i:i + 1]
        i += 1
        tw = dm[:, i:i + 1]
        i += 1
        th = dm[:, i:i + 1]
        i += 1
        sn = dm[:, i:i + 1]
        i += 1
        cs = dm[:, i:i + 1]
        i += 1
        obj = dm[:, i:i + 1]
        i += 1
        cls = dm[:, i:] if self.nc > 0 else dm.new_zeros(B, 0, H, W)
        return dict(tx=tx, ty=ty, tw=tw, th=th, sin=sn, cos=cs, obj=obj, cls=cls)

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
    @torch.no_grad()
    def decode_obb_from_pyramids(
            self,
            det_maps: list,  # list of tensors [B, 7+nc, h, w]
            imgs: torch.Tensor,  # [B,3,H,W] or list of sizes
            *,
            strides=None,
            conf_thres=None,  # legacy name
            score_thr=None,  # evaluator's arg
            iou_thres=0.50,
            max_det=300,
            multi_label=False,
            agnostic=False,
            use_nms=True,
            **kwargs,
    ):
        """
        Decode [tx,ty,tw,th,sin,cos,obj,cls...] into (cx,cy,w,h,angle,score,cls) per image.
        - angles are returned in degrees in [-90,90) for mmcv.ops.nms_rotated
        - score = sigmoid(obj) * sigmoid(cls)  (or obj only if nc==1)
        """
        # ---- config / safety -----------------------------------------------------
        s_list = tuple(self.strides if strides is None else strides)
        assert len(det_maps) == len(s_list), f"len(pyramids)={len(det_maps)} vs strides={len(s_list)}"

        # evaluator sometimes passes None/torch tensors here; sanitize
        Himg, Wimg = (int(imgs.shape[-2]), int(imgs.shape[-1])) if torch.is_tensor(imgs) else imgs
        thr = float(score_thr) if score_thr is not None else (float(conf_thres) if conf_thres is not None else 0.25)
        iou = float(iou_thres)

        try:
            from mmcv.ops import nms_rotated  # available in your environment
            has_rot_nms = True
        except Exception:
            has_rot_nms = False

        # debug (avoid formatting tensors)
        if 'rank0' in kwargs or True:
            # only print once per call
            print(f"[DECODE ASSERT] classique | nc={self.nc} conf_thr={thr:.3f} use_nms={bool(use_nms)} iou={iou:.2f}")

        B = det_maps[0].shape[0]
        out_per_image = []

        for b in range(B):
            all_bboxes = []
            all_scores = []
            all_labels = []

            for lvl, (dm, s) in enumerate(zip(det_maps, s_list)):
                # shapes / grid
                _, C, h, w = dm.shape
                mp = self._split_map(dm)  # (B,1,h,w) etc.

                tx = mp['tx'][b]  # (1,h,w)
                ty = mp['ty'][b]
                tw = mp['tw'][b]
                th = mp['th'][b]
                sn = mp['sin'][b]
                cs = mp['cos'][b]
                ob = mp['obj'][b]  # (1,h,w)
                cl = mp['cls'][b]  # (nc,h,w) or (0,h,w)

                # grid centers in cell units
                gx, gy = self._make_grid(h, w, device=dm.device, dtype=tx.dtype)  # (1,1,h,w)
                # decode centers to pixels
                cx = (gx + torch.sigmoid(tx)).squeeze(0) * float(s)  # (h,w)
                cy = (gy + torch.sigmoid(ty)).squeeze(0) * float(s)

                # width/height in pixels
                pw = self._safe_exp(tw).squeeze(0) * float(s)  # (h,w)
                ph = self._safe_exp(th).squeeze(0) * float(s)

                # angle in radians -> degrees in [-90,90)
                ang = torch.atan2(sn, cs).squeeze(0)  # (h,w) rad
                ang_deg = (ang * 180.0 / math.pi)
                # wrap to [-90, 90)
                ang_deg = ((ang_deg + 90.0) % 180.0) - 90.0

                # clamp sizes to avoid degenerate boxes
                max_wh = float(2.0 * max(Himg, Wimg))
                pw = pw.clamp_(min=1.0, max=max_wh)
                ph = ph.clamp_(min=1.0, max=max_wh)

                # scores
                obj = torch.sigmoid(ob).squeeze(0)  # (h,w)
                if self.nc > 0:
                    cls_prob = torch.sigmoid(cl)  # (nc,h,w)
                    if multi_label and self.nc > 1:
                        # keep all classes above threshold per location
                        cls_prob_flat = cls_prob.reshape(self.nc, -1)  # (nc, h*w)
                        obj_flat = obj.reshape(-1).unsqueeze(0).expand_as(cls_prob_flat)
                        conf_flat = (cls_prob_flat * obj_flat)  # (nc, h*w)
                        keep = conf_flat > thr
                        if keep.any():
                            ys, xs = torch.div(torch.nonzero(keep, as_tuple=False)[:, 1], w, rounding_mode='floor'), \
                                torch.nonzero(keep, as_tuple=False)[:, 1] % w
                            kcls = torch.nonzero(keep, as_tuple=False)[:, 0]
                            sel = conf_flat[keep]
                            bb = torch.stack([cx[ys, xs], cy[ys, xs], pw[ys, xs], ph[ys, xs], ang_deg[ys, xs]], dim=1)
                            all_bboxes.append(bb)
                            all_scores.append(sel)
                            all_labels.append(kcls)
                    else:
                        # best class per cell
                        cls_max, cls_idx = cls_prob.max(dim=0)  # (h,w)
                        conf = (cls_max * obj)
                        keep = conf > thr
                        if keep.any():
                            ys, xs = torch.where(keep)
                            bb = torch.stack([cx[ys, xs], cy[ys, xs], pw[ys, xs], ph[ys, xs], ang_deg[ys, xs]], dim=1)
                            all_bboxes.append(bb)
                            all_scores.append(conf[ys, xs])
                            all_labels.append(cls_idx[ys, xs])
                else:
                    # nc==0 case: score = obj only
                    conf = obj
                    keep = conf > thr
                    if keep.any():
                        ys, xs = torch.where(keep)
                        bb = torch.stack([cx[ys, xs], cy[ys, xs], pw[ys, xs], ph[ys, xs], ang_deg[ys, xs]], dim=1)
                        all_bboxes.append(bb)
                        all_scores.append(conf[ys, xs])
                        all_labels.append(torch.zeros_like(conf[ys, xs], dtype=torch.long))

            # collate per image
            if len(all_bboxes) == 0:
                out_per_image.append(torch.empty(0, 7, device=imgs.device))
                continue

            bboxes = torch.cat(all_bboxes, dim=0)  # (N,5) cx,cy,w,h,angle(deg)
            scores = torch.cat(all_scores, dim=0)  # (N,)
            labels = torch.cat(all_labels, dim=0)  # (N,)

            if bboxes.numel() == 0:
                out_per_image.append(torch.empty(0, 7, device=imgs.device))
                continue

            # NMS (class-agnostic if requested)
            if use_nms and has_rot_nms and bboxes.shape[0] > 0:
                dets_keep = []
                if agnostic or self.nc <= 1:
                    dets, keep_idx = nms_rotated(bboxes, scores, iou)
                    # dets: (M, 6) [cx,cy,w,h,angle,score]
                    lbl = labels[keep_idx]
                    dets_keep = torch.cat([dets[:, :5], dets[:, 5:6], lbl.float().unsqueeze(1)], dim=1)  # (M,7)
                else:
                    # per-class NMS
                    for c in range(self.nc):
                        idx = (labels == c)
                        if idx.any():
                            dets, keep_idx = nms_rotated(bboxes[idx], scores[idx], iou)
                            if dets.numel() > 0:
                                lbl = torch.full((dets.shape[0], 1), float(c), device=dets.device)
                                dets_keep.append(torch.cat([dets[:, :5], dets[:, 5:6], lbl], dim=1))
                    dets_keep = torch.cat(dets_keep, dim=0) if len(dets_keep) else torch.empty(0, 7,
                                                                                               device=bboxes.device)
                # sort by score desc and limit max_det
                if dets_keep.numel() > 0:
                    dets_keep = dets_keep[torch.argsort(dets_keep[:, 5], descending=True)]
                    dets_keep = dets_keep[:max_det]
                out_per_image.append(dets_keep)
            else:
                # no NMS: just stack and top-k by score
                dets = torch.cat([bboxes, scores.unsqueeze(1), labels.float().unsqueeze(1)], dim=1)  # (N,7)
                dets = dets[torch.argsort(dets[:, 5], descending=True)]
                dets = dets[:max_det]
                out_per_image.append(dets)

        return out_per_image






