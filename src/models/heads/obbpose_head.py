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

    def __init__(self, ch: Tuple[int, int, int], num_classes: int, reg_max: int = 16) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self.nbins = self.reg_max + 1

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
            det_maps: List[torch.Tensor],
            imgs: torch.Tensor,
            *,
            strides: Tuple[int, int, int] = (8, 16, 32),
            conf_thres: Optional[float] = None,
            score_thr: Optional[float] = None,
            iou_thres: float = 0.5,
            max_det: int = 300,
            multi_label: bool = False,
            agnostic: bool = False,
            use_nms: bool = False,
            pre_nms_topk: Optional[int] = None,
            fallback_topk_per_level: int = 200,  # NEW: used if nothing passes threshold
            **kwargs,
    ):
        """
        DFL-only decoder.
        Channel layout per level (C = 2 + (reg_max+1) + (reg_max+1) + 1 + 1 + nc):
          [ tx, ty, dflw(nbins), dflh(nbins), ang_logit, obj, (cls...)]
        Angle is single-logit mapped to [-pi/2, pi/2). Returns per-image dicts:
          {"boxes": (N,5 [cx,cy,w,h,ang]), "scores": (N,), "labels": (N,)}
        """
        import math
        device = imgs.device
        B, _, Himg, Wimg = imgs.shape

        # permissive default threshold (avoid 'pred empty' at early epochs)
        thr = conf_thres if conf_thres is not None else score_thr
        thr = 0.0 if thr is None else float(thr)

        # DFL bins
        if getattr(self, "reg_max", None) is None and getattr(self, "nbins", None) is None:
            raise RuntimeError("DFL decode requires self.reg_max (or self.nbins) on the head.")
        reg_max = int(self.reg_max) if getattr(self, "reg_max", None) is not None else int(self.nbins) - 1
        assert reg_max >= 0
        nbins = reg_max + 1

        # one-time friendly log
        if not hasattr(self, "_decode_once_print"):
            self._decode_once_print = True
            try:
                import torch.distributed as dist
                main_rank = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
            except Exception:
                main_rank = True
            if main_rank:
                print(
                    f"[DECODE ASSERT] DFL-only | reg_max={reg_max} nbins={nbins} conf_thres={thr:.3f} use_nms={use_nms} iou={iou_thres:.2f}")

        # helpers
        def make_grid(h, w, dev):
            yv, xv = torch.meshgrid(torch.arange(h, device=dev),
                                    torch.arange(w, device=dev), indexing="ij")
            return xv.view(1, 1, h, w), yv.view(1, 1, h, w)

        def ang_from_logit(z: torch.Tensor) -> torch.Tensor:
            # map sigmoid(z) to [-pi/2, pi/2)
            return z.sigmoid() * math.pi - (math.pi / 2.0)

        def le90(w: torch.Tensor, h: torch.Tensor, ang: torch.Tensor):
            swap = w < h
            w2 = torch.where(swap, h, w)
            h2 = torch.where(swap, w, h)
            ang2 = torch.where(swap, ang + math.pi / 2.0, ang)
            # wrap back to [-pi/2, pi/2)
            ang2 = torch.remainder(ang2 + math.pi / 2.0, math.pi) - math.pi / 2.0
            return w2, h2, ang2

        def nms_xyxy_fallback(xyxy: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
            try:
                from torchvision.ops import nms
                return nms(xyxy, scores, iou_thr)
            except Exception:
                x1 = xyxy[:, 0]
                y1 = xyxy[:, 1]
                x2 = xyxy[:, 2]
                y2 = xyxy[:, 3]
                areas = (x2 - x1).clamp_min_(0) * (y2 - y1).clamp_min_(0)
                order = scores.argsort(descending=True)
                keep = []
                while order.numel() > 0:
                    i = order[0].item()
                    keep.append(i)
                    if order.numel() == 1:
                        break
                    rest = order[1:]
                    xx1 = torch.maximum(x1[i], x1[rest])
                    yy1 = torch.maximum(y1[i], y1[rest])
                    xx2 = torch.minimum(x2[i], x2[rest])
                    yy2 = torch.minimum(y2[i], y2[rest])
                    inter = (xx2 - xx1).clamp_min_(0) * (yy2 - yy1).clamp_min_(0)
                    iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
                    rest = rest[iou <= iou_thr]
                    order = rest
                return torch.as_tensor(keep, device=xyxy.device, dtype=torch.long)

        out_boxes = [[] for _ in range(B)]
        out_scores = [[] for _ in range(B)]
        out_labels = [[] for _ in range(B)]

        L = len(det_maps)
        max_det = int(max(1, max_det))
        per_level_topk = pre_nms_topk if (pre_nms_topk is not None) else max(1, max_det * 5 // max(L, 1))

        for lvl, (dm, s) in enumerate(zip(det_maps, strides)):
            assert dm.dim() == 4, "det map must be (B,C,H,W)"
            Bb, C, H, W = dm.shape
            assert Bb == B

            base = 2 + nbins + nbins + 1 + 1  # tx,ty + dflw + dflh + ang + obj
            if C < base:
                raise RuntimeError(f"Level {lvl}: channels={C} < expected base={base}.")
            nc = C - base  # may be zero

            idx = 0
            tx = dm[:, idx:idx + 1]
            idx += 1
            ty = dm[:, idx:idx + 1]
            idx += 1
            dflw = dm[:, idx:idx + nbins]
            idx += nbins
            dflh = dm[:, idx:idx + nbins]
            idx += nbins
            ang_logit = dm[:, idx:idx + 1]
            idx += 1
            obj = dm[:, idx:idx + 1]
            idx += 1
            cls = dm[:, idx:] if nc > 0 else None

            # one-time stride print (first level only)
            if not hasattr(self, "_decode_stride_once"):
                self._decode_stride_once = True
                s_h = int(round(Himg / H)) if H > 0 else 1
                s_w = int(round(Wimg / W)) if W > 0 else 1
                s_auto = s_w if s_w == s_h else int(round((Himg / max(1, H) + Wimg / max(1, W)) * 0.5))
                print(f"[DECODE ASSERT] level#{lvl} fmap={H}x{W}  stride={s}  auto_stride={s_auto}")

            gx, gy = make_grid(H, W, device)
            N = H * W

            obj_prob = obj.sigmoid().view(B, -1)  # (B, N)

            for b in range(B):
                keep = (obj_prob[b] > thr).nonzero(as_tuple=False).squeeze(1)

                # NEW: fallback if threshold prunes everything on this level
                if keep.numel() == 0:
                    topk = min(fallback_topk_per_level, N)
                    _, keep = obj_prob[b].topk(topk, largest=True, sorted=False)

                # optionally cap per level pre-NMS
                if keep.numel() > per_level_topk:
                    topv, topi = torch.topk(obj_prob[b].index_select(0, keep), k=per_level_topk, largest=True,
                                            sorted=False)
                    keep = keep.index_select(0, topi)

                ix = (keep % W).to(tx.dtype)
                iy = (keep // W).to(tx.dtype)

                # centers
                sx = tx[b, 0].view(-1).index_select(0, keep).sigmoid()
                sy = ty[b, 0].view(-1).index_select(0, keep).sigmoid()
                cx = (ix + sx) * float(s)
                cy = (iy + sy) * float(s)

                # DFL expected width/height
                dflw_k = dflw[b].reshape(nbins, N).index_select(1, keep).float()
                dflh_k = dflh[b].reshape(nbins, N).index_select(1, keep).float()
                bin_idx = torch.arange(nbins, device=device, dtype=dflw_k.dtype).view(nbins, 1)
                pw = (torch.softmax(dflw_k, dim=0) * bin_idx).sum(0) * float(s)
                ph = (torch.softmax(dflh_k, dim=0) * bin_idx).sum(0) * float(s)

                # angle
                ang = ang_from_logit(ang_logit[b, 0].view(-1).index_select(0, keep)).to(dm.dtype)

                # LE-90 canonicalisation
                pw, ph, ang = le90(pw.to(dm.dtype), ph.to(dm.dtype), ang)

                # class scores
                obj_k = obj_prob[b].index_select(0, keep)
                if cls is None or cls.shape[1] == 0:
                    scores = obj_k
                    labels = torch.zeros_like(scores, dtype=torch.long, device=device)
                elif multi_label:
                    cls_b = cls[b].reshape(nc, N)
                    s_mat = (cls_b.index_select(1, keep).sigmoid() * obj_k.view(1, -1))
                    c_idx, p_idx = (s_mat > thr).nonzero(as_tuple=True)
                    if p_idx.numel() == 0:
                        # fall back to best class per anchor
                        cls_scores, cls_labels = s_mat.max(dim=0)
                        scores = cls_scores
                        labels = cls_labels.to(torch.long)
                    else:
                        scores = s_mat[c_idx, p_idx]
                        labels = c_idx.to(torch.long)
                        cx = cx.index_select(0, p_idx)
                        cy = cy.index_select(0, p_idx)
                        pw = pw.index_select(0, p_idx)
                        ph = ph.index_select(0, p_idx)
                        ang = ang.index_select(0, p_idx)
                else:
                    cls_b = cls[b].reshape(nc, N).index_select(1, keep).sigmoid()
                    cls_scores, cls_labels = cls_b.max(dim=0)
                    scores = cls_scores * obj_k
                    labels = cls_labels.to(torch.long)
                    if agnostic:
                        labels.zero_()

                boxes = torch.stack([cx, cy, pw, ph, ang], dim=1)

                # NMS
                if use_nms and boxes.numel():
                    try:
                        from mmcv.ops import nms_rotated as mmcv_nms_rotated
                        rb = torch.cat([boxes[:, :4], (boxes[:, 4] * 180.0 / math.pi).unsqueeze(1)], dim=1)
                        keep_idx = mmcv_nms_rotated(rb, scores, iou_thres, score_threshold=float(thr))[1]
                    except Exception:
                        xyxy = torch.stack([
                            boxes[:, 0] - boxes[:, 2] * 0.5,
                            boxes[:, 1] - boxes[:, 3] * 0.5,
                            boxes[:, 0] + boxes[:, 2] * 0.5,
                            boxes[:, 1] + boxes[:, 3] * 0.5,
                        ], dim=1)
                        keep_idx = nms_xyxy_fallback(xyxy, scores, iou_thres)
                    boxes = boxes.index_select(0, keep_idx)
                    scores = scores.index_select(0, keep_idx)
                    labels = labels.index_select(0, keep_idx)

                # cap per-level before concat
                if boxes.shape[0] > per_level_topk:
                    topv, topi = torch.topk(scores, k=per_level_topk, largest=True, sorted=False)
                    boxes = boxes.index_select(0, topi)
                    labels = labels.index_select(0, topi)
                    scores = topv

                out_boxes[b].append(boxes)
                out_scores[b].append(scores)
                out_labels[b].append(labels)

        # concat per image & global cap
        results = []
        for b in range(B):
            if out_boxes[b]:
                boxes = torch.cat(out_boxes[b], dim=0)
                scores = torch.cat(out_scores[b], dim=0)
                labels = torch.cat(out_labels[b], dim=0)
                if boxes.shape[0] > max_det:
                    topv, topi = torch.topk(scores, k=max_det, largest=True, sorted=False)
                    boxes = boxes.index_select(0, topi)
                    labels = labels.index_select(0, topi)
                    scores = topv
            else:
                boxes = torch.zeros((0, 5), device=device)
                scores = torch.zeros((0,), device=device)
                labels = torch.zeros((0,), device=device, dtype=torch.long)
            results.append({"boxes": boxes, "scores": scores, "labels": labels})
        return results





