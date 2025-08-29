"""
Modified OBBPoseHead for YOLO‑style oriented bounding box detection.

This head predicts offsets, distributional width/height bins (DFL), a single
angle logit, objectness and class logits at three scales (P3, P4, P5).
The implementation borrows the core ideas from Ultralytics' YOLO models
but exposes a `reg_max` parameter to control the number of bins used for
width/height regression.  It also initialises the objectness bias to a
small negative value to encourage low confidences at the start of training.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import math
import torch
import torch.nn as nn


class OBBPoseHead(nn.Module):
    """
    YOLO‑style OBB + tiny keypoint head.

    Each level (P3, P4, P5) outputs a tensor with channels ordered as:

      [ tx, ty,
        dfl_w[0..reg_max], dfl_h[0..reg_max],
        ang_logit, obj, (cls0..cls{nc-1}) ]

    The offsets `tx, ty` are raw regression values (passed through sigmoid
    during decoding).  The width and height are predicted as a discrete
    probability distribution across `reg_max+1` bins; the decoder computes
    the expected value of this distribution and multiplies by the stride to
    recover the bounding box size in pixels.  A single angle logit is
    transformed to a rotation in the range [-pi/2, pi/2) during decoding
    and then canonicalised to the LE‑90 representation (w ≥ h).  The
    objectness and class channels are raw logits.
    """

    def __init__(self, ch: Tuple[int, int, int], num_classes: int, reg_max: int = 8) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        # The largest bin index; total bins = reg_max + 1
        self.reg_max = int(reg_max)
        self.nbins = self.reg_max + 1

        # cached bins for DFL expected value calculation
        self.register_buffer(
            "dfl_bins",
            torch.arange(self.nbins, dtype=torch.float32).view(1, self.nbins, 1, 1),
            persistent=False,
        )

        c3, c4, c5 = [int(x) for x in ch]
        # output channels: tx, ty, DFL(w), DFL(h), ang, obj, classes
        det_out = 2 + 2 * self.nbins + 1 + 1 + self.num_classes
        kpt_out = 3  # (kpx, kpy, kpscore) – kept for compatibility

        # P3 detection head
        self.det3 = nn.Sequential(
            nn.Conv2d(c3, c3, 3, 1, 1),
            nn.BatchNorm2d(c3),
            # Avoid in‑place activation to prevent autograd errors.
            nn.SiLU(),
            nn.Conv2d(c3, det_out, 1, 1, 0)
        )
        # P4 detection head
        self.det4 = nn.Sequential(
            nn.Conv2d(c4, c4, 3, 1, 1),
            nn.BatchNorm2d(c4),
            # Avoid in‑place activation to prevent autograd errors.
            nn.SiLU(),
            nn.Conv2d(c4, det_out, 1, 1, 0)
        )
        # P5 detection head
        self.det5 = nn.Sequential(
            nn.Conv2d(c5, c5, 3, 1, 1),
            nn.BatchNorm2d(c5),
            # Avoid in‑place activation to prevent autograd errors.
            nn.SiLU(),
            nn.Conv2d(c5, det_out, 1, 1, 0)
        )

        # Tiny keypoint heads (unused in most setups but left for extensibility)
        self.kp3 = nn.Sequential(
            nn.Conv2d(c3, c3, 3, 1, 1),
            nn.BatchNorm2d(c3),
            # Avoid in‑place activation to prevent autograd errors.
            nn.SiLU(),
            nn.Conv2d(c3, kpt_out, 1, 1, 0)
        )
        self.kp4 = nn.Sequential(
            nn.Conv2d(c4, c4, 3, 1, 1),
            nn.BatchNorm2d(c4),
            # Avoid in‑place activation to prevent autograd errors.
            nn.SiLU(),
            nn.Conv2d(c4, kpt_out, 1, 1, 0)
        )
        self.kp5 = nn.Sequential(
            nn.Conv2d(c5, c5, 3, 1, 1),
            nn.BatchNorm2d(c5),
            # Avoid in‑place activation to prevent autograd errors.
            nn.SiLU(),
            nn.Conv2d(c5, kpt_out, 1, 1, 0)
        )

        # lightweight grid cache for decoding
        self._grid_cache: dict[tuple[torch.device, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        # one‑off debug flags
        self._assert_once = False
        self._printed_ang_norm = False
        self._dbg_decode_once = False

        # initialise biases: encourage low objectness at start
        self._init_det_bias(self.det3)
        self._init_det_bias(self.det4)
        self._init_det_bias(self.det5)

    def _init_det_bias(self, mod: nn.Sequential) -> None:
        """Initialise the objectness bias to a small negative value."""
        final = mod[-1]
        assert isinstance(final, nn.Conv2d)
        with torch.no_grad():
            b = final.bias
            if b is None or b.numel() == 0:
                final.bias = nn.Parameter(torch.zeros_like(final.weight[:, 0, 0, 0]))
                b = final.bias
            b.zero_()
            # channel layout: [tx, ty, dflw(nb), dflh(nb), ang, obj, classes]
            obj_idx = 2 + 2 * self.nbins + 1  # index of objectness channel
            if b.numel() > obj_idx:
                b[obj_idx] = -3.0  # ~logit(1%)

    # -------------------------- forward --------------------------
    def forward(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Produce detection and keypoint maps at three scales.

        Args:
            feats: list of feature maps [P3, P4, P5].
        Returns:
            det_maps: list of detection maps [(B, C, H, W) × 3]
            kpt_maps: list of keypoint maps [(B, 3, H, W) × 3]
        """
        assert isinstance(feats, (list, tuple)) and len(feats) == 3, (
            "OBBPoseHead.forward expects 3 feature maps [P3,P4,P5]")
        p3, p4, p5 = feats
        det_maps = [self.det3(p3), self.det4(p4), self.det5(p5)]
        kpt_maps = [self.kp3(p3), self.kp4(p4), self.kp5(p5)]

        # one‑time sanity check on channel count
        if not self._assert_once:
            expected = 2 + 2 * self.nbins + 1 + 1 + self.num_classes
            for i, lvl in enumerate(("P3", "P4", "P5")):
                C = int(det_maps[i].shape[1])
                assert C == expected, (
                    f"OBBPoseHead({lvl}) channels={C}, expected {expected}="
                    f"[tx,ty,dflw({self.nbins}),dflh({self.nbins}),ang,obj,(cls...)]"
                )
            self._assert_once = True
        # print single‑logit angle notice once
        if not self._printed_ang_norm:
            print("[ASSERT] angle uses single‑logit (YOLO‑style).")
            self._printed_ang_norm = True
        return det_maps, kpt_maps

    # -------------------------- decoding --------------------------
    @staticmethod
    def _le90(w: torch.Tensor, h: torch.Tensor, ang: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ensure width ≥ height and rotate angle by 90° when swapped."""
        swap = w < h
        w2 = torch.where(swap, h, w)
        h2 = torch.where(swap, w, h)
        ang2 = torch.where(swap, ang + math.pi / 2.0, ang)
        # wrap angle back to [-pi/2, pi/2)
        ang2 = torch.remainder(ang2 + math.pi / 2.0, math.pi) - math.pi / 2.0
        return w2, h2, ang2

    def _ev_from_dfl(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute expected value from DFL logits along the channel dimension."""
        probs = torch.softmax(logits, dim=1)
        bins = self.dfl_bins.to(device=logits.device, dtype=logits.dtype, non_blocking=True)
        return (probs * bins).sum(dim=1, keepdim=True)

    @torch.no_grad()
    def decode_obb_from_pyramids(
        self,
        det_maps: List[torch.Tensor],
        imgs: torch.Tensor,
        *,
        strides: Tuple[int, int, int] = (8, 16, 32),
        score_thr: float = 0.01,
        iou_thres: float = 0.5,
        max_det: int = 300,
        use_nms: bool = True,
        multi_label: bool = False,
        **kwargs,
    ) -> List[dict[str, torch.Tensor]]:
        """
        Decode detection maps into oriented boxes, scores and labels.

        This function applies a sigmoid to offsets and objectness, softmax to
        DFL bins, computes the expected width/height in stride units, scales
        them to pixel space, applies a sigmoid to the angle logit to map it
        to [-pi/2, pi/2), canonicalises the boxes to LE‑90 and optionally
        applies NMS.  It returns a list of dictionaries per batch item.
        """
        B, _, Himg, Wimg = imgs.shape
        device = imgs.device
        nbins = self.nbins
        results = []

        # one‑time debug print
        if not hasattr(self, "_decode_once_print"):
            self._decode_once_print = True
            print(f"[DECODE ASSERT] DFL decode | reg_max={self.reg_max} nbins={self.nbins} score_thr={score_thr:.3f} use_nms={use_nms} iou={iou_thres:.2f}")

        for b in range(B):
            boxes_all = []
            scores_all = []
            labels_all = []
            for lvl, (dm, s) in enumerate(zip(det_maps, strides)):
                _, C, H, W = dm.shape
                # split channels
                idx = 0
                tx = dm[b, idx:idx + 1]; idx += 1
                ty = dm[b, idx:idx + 1]; idx += 1
                dflw = dm[b, idx:idx + nbins]; idx += nbins
                dflh = dm[b, idx:idx + nbins]; idx += nbins
                ang_logit = dm[b, idx:idx + 1]; idx += 1
                obj = dm[b, idx:idx + 1]; idx += 1
                cls = dm[b, idx:] if C > idx else None

                # flatten spatial dims
                N = H * W
                obj_prob = obj.sigmoid().view(-1)
                # select indices above threshold or fallback topk
                keep = (obj_prob > score_thr).nonzero(as_tuple=False).squeeze(1)
                if keep.numel() == 0:
                    # fallback to highest‑confidence detections per level
                    topk = min(200, N)
                    _, keep = obj_prob.topk(topk, largest=True, sorted=False)

                # compute grid indices
                iy = (keep // W).to(tx.dtype)
                ix = (keep % W).to(tx.dtype)

                # decode centre
                sx = tx[0].view(-1).index_select(0, keep).sigmoid()
                sy = ty[0].view(-1).index_select(0, keep).sigmoid()
                cx = (ix + sx) * float(s)
                cy = (iy + sy) * float(s)

                # decode width/height via expected value
                dflw_k = dflw.reshape(nbins, N).index_select(1, keep).float()
                dflh_k = dflh.reshape(nbins, N).index_select(1, keep).float()
                bin_idx = torch.arange(nbins, device=device, dtype=dflw_k.dtype).view(nbins, 1)
                pw = (torch.softmax(dflw_k, dim=0) * bin_idx).sum(0) * float(s)
                ph = (torch.softmax(dflh_k, dim=0) * bin_idx).sum(0) * float(s)

                # decode angle
                ang = ang_logit.view(-1).index_select(0, keep).sigmoid() * math.pi - (math.pi / 2.0)

                # canonicalise
                pw, ph, ang = self._le90(pw.to(obj.dtype), ph.to(obj.dtype), ang)

                # decode class and score
                obj_k = obj_prob.index_select(0, keep)
                if cls is None or cls.shape[0] == 0:
                    scores = obj_k
                    labels = torch.zeros_like(scores, dtype=torch.long, device=device)
                elif multi_label:
                    cls_b = cls.sigmoid().view(self.num_classes, N).index_select(1, keep)
                    s_mat = cls_b * obj_k.unsqueeze(0)
                    c_idx, p_idx = (s_mat > score_thr).nonzero(as_tuple=True)
                    if p_idx.numel() == 0:
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
                    cls_b = cls.sigmoid().view(self.num_classes, N).index_select(1, keep)
                    cls_scores, cls_labels = cls_b.max(dim=0)
                    scores = cls_scores * obj_k
                    labels = cls_labels.to(torch.long)

                # accumulate
                boxes = torch.stack([cx, cy, pw, ph, ang], dim=1)
                boxes_all.append(boxes)
                scores_all.append(scores)
                labels_all.append(labels)

            # concatenate across levels
            if boxes_all:
                boxes = torch.cat(boxes_all, dim=0)
                scores = torch.cat(scores_all, dim=0)
                labels = torch.cat(labels_all, dim=0)
                # NMS if requested
                if use_nms and boxes.numel() > 0:
                    # Use mmcv.rotated NMS in radian space; fallback to axis‑aligned NMS if unavailable.
                    try:
                        from mmcv.ops import nms_rotated as mmcv_nms_rotated
                        # mmcv expects angles in radians and returns kept detections and indices.
                        # Note: we pass clockwise=False to match the dataset's CCW angle convention.
                        dets, keep_idx = mmcv_nms_rotated(boxes, scores, iou_thres, labels=None, clockwise=False)
                        boxes = boxes.index_select(0, keep_idx)
                        scores = scores.index_select(0, keep_idx)
                        labels = labels.index_select(0, keep_idx)
                    except Exception:
                        # fallback to AABB NMS using torchvision
                        x1 = boxes[:, 0] - boxes[:, 2] * 0.5
                        y1 = boxes[:, 1] - boxes[:, 3] * 0.5
                        x2 = boxes[:, 0] + boxes[:, 2] * 0.5
                        y2 = boxes[:, 1] + boxes[:, 3] * 0.5
                        from torchvision.ops import nms
                        abox = torch.stack([x1, y1, x2, y2], dim=1)
                        keep_idx = nms(abox, scores, iou_thres)
                        boxes = boxes.index_select(0, keep_idx)
                        scores = scores.index_select(0, keep_idx)
                        labels = labels.index_select(0, keep_idx)
                # cap number of detections per image
                if boxes.shape[0] > max_det:
                    topk_idx = scores.topk(max_det, largest=True, sorted=False).indices
                    boxes = boxes.index_select(0, topk_idx)
                    scores = scores.index_select(0, topk_idx)
                    labels = labels.index_select(0, topk_idx)
            else:
                boxes = torch.zeros((0, 5), device=device)
                scores = torch.zeros((0,), device=device)
                labels = torch.zeros((0,), dtype=torch.long, device=device)

            results.append({"boxes": boxes, "scores": scores, "labels": labels})
        return results