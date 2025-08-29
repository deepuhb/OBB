"""
Loss functions for the YOLO‑style OBB + keypoint detector.

This module defines a criterion that supports distributional focal loss (DFL) for
width and height regression, a single‑logit angle regression, objectness and
classification losses, as well as an optional keypoint regression loss.  The
criterion follows a simplified version of Ultralytics' approach: each ground
truth object is assigned to one of three feature map levels based on its
maximum side length, and to a small neighbourhood of grid cells around its
centre.  For positive cells, the loss regresses the centre offsets, box
dimensions and angle, while supervising objectness and classification.  For
negative cells, only objectness and classification are penalised.  The DFL
component supervises the probability distribution over discrete bins used to
represent widths and heights; the expected value of this distribution is
decoded during training and inference.

The criterion is designed to pair with the ``OBBPoseHead`` defined in
``src/models/heads/obbpose_head.py``.  It accepts detection feature maps from
each level and a list of per‑image targets containing oriented boxes,
class labels and optional keypoints.  The returned loss is the weighted sum
of its individual components along with a dictionary for logging.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TDOBBWKpt1Criterion(nn.Module):
    """Loss for distributional OBB + keypoint detection.

    This criterion supervises oriented bounding boxes predicted with a single-logit
    angle and distributional focal loss (DFL) for width/height.  It additionally
    supports configurable positive-sample assignment and an optional IoU-based
    penalty.  Ground-truth objects are routed to one of three pyramid levels
    based on their longest side and assigned to a Manhattan neighbourhood of
    grid cells.  For each positive cell, we regress centre offsets, sizes and
    angle, compute DFL soft targets for the width/height distributions, and
    supervise objectness and (optional) class scores.  Negatives receive
    objectness/class penalties on all unassigned cells.  Keypoint regression
    is handled via a tiny crop head when ``use_kpt`` is True.

    Args:
        num_classes: number of object classes (>=1).
        reg_max: maximum discrete bin index for DFL (bin count = reg_max + 1).
        strides: tuple of strides for each pyramid level (e.g. (8,16,32)).
        lambda_box: weight for centre (x,y) and size (w,h) regression losses.
        lambda_obj: weight for objectness loss.
        lambda_cls: weight for classification loss.
        lambda_ang: weight for angle regression loss.
        lambda_dfl: weight for DFL distribution loss.
        lambda_kpt: weight for keypoint regression loss.
        level_boundaries: boundaries on the longest side for assigning GT to levels.
        neighbor_cells: if True, assign each GT to neighbouring grid cells within
            ``neighbor_range``; if False, only the closest cell is used.
        neighbor_range: Manhattan distance around the centre cell for positive
            samples; ignored when ``neighbor_cells`` is False.
        use_kpt: whether to supervise keypoint predictions.
        lambda_iou: additional weight for an IoU-based penalty term added to
            the box regression loss.  Set to 0.0 to disable.
    """

    def __init__(
        self,
        num_classes: int,
        reg_max: int = 8,
        strides: Tuple[int, int, int] = (8, 16, 32),
        lambda_box: float = 5.0,
        lambda_obj: float = 1.0,
        lambda_cls: float = 0.5,
        lambda_ang: float = 0.5,
        lambda_dfl: float = 0.5,
        lambda_kpt: float = 2.0,
        level_boundaries: Tuple[float, float] = (32.0, 64.0),
        neighbor_cells: bool = True,
        neighbor_range: int = 1,
        use_kpt: bool = True,
        lambda_iou: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self.nbins = self.reg_max + 1
        assert len(strides) == 3, "Expect strides for P3, P4, P5"
        self.strides = tuple(float(s) for s in strides)
        self.lambda_box = float(lambda_box)
        self.lambda_obj = float(lambda_obj)
        self.lambda_cls = float(lambda_cls)
        self.lambda_ang = float(lambda_ang)
        self.lambda_dfl = float(lambda_dfl)
        self.lambda_kpt = float(lambda_kpt)
        # weight for IoU penalty (zero disables)
        self.lambda_iou = float(lambda_iou)
        # routing and assignment parameters
        self.level_boundaries = tuple(float(x) for x in level_boundaries)
        self.neighbor_cells = bool(neighbor_cells)
        self.neighbor_range = int(neighbor_range)
        self.use_kpt = bool(use_kpt)
        # bins tensor for DFL expected value and soft targets
        self.register_buffer(
            "bins",
            torch.arange(self.nbins, dtype=torch.float32).view(1, self.nbins),
            persistent=False,
        )

    def _get_level(self, max_side: float) -> int:
        """Return the index of the feature map level for a given object size."""
        b0, b1 = self.level_boundaries
        if max_side < b0:
            return 0
        elif max_side < b1:
            return 1
        else:
            return 2

    @staticmethod
    def _le90(w: torch.Tensor, h: torch.Tensor, ang: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Canonicalise width ≥ height and wrap angle into [-pi/2, pi/2)."""
        swap = w < h
        w2 = torch.where(swap, h, w)
        h2 = torch.where(swap, w, h)
        ang2 = torch.where(swap, ang + math.pi / 2.0, ang)
        ang2 = torch.remainder(ang2 + math.pi / 2.0, math.pi) - math.pi / 2.0
        return w2, h2, ang2

    def forward(
        self,
        det_maps: List[torch.Tensor],
        kpt_maps: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the total loss and log individual components.

        Args:
            det_maps: list of detection feature maps at three scales.  Each map
                should have shape (B, C, H, W) with channels ordered as
                [tx, ty, dfl_w(nbins), dfl_h(nbins), angle_logit, obj, classes].
            kpt_maps: list of keypoint maps at three scales.  Each map has
                shape (B, 3, H, W) with channels [kpx, kpy, kp_score].  These
                are ignored if ``use_kpt`` is False or no keypoints are provided.
            targets: list of dictionaries per image.  Keys:
              "boxes"  – Tensor shape (N, 5) with (cx, cy, w, h, theta) in pixels.
              "labels" – Tensor shape (N,) with class indices in [0, num_classes).
              "keypoints" – Optional Tensor shape (N, 2) with (kx, ky) in pixels.

        Returns:
            total_loss: weighted sum of all components.
            loss_dict: dictionary of individual loss components for logging.
        """
        device = det_maps[0].device
        batch_size = det_maps[0].shape[0]
        assert batch_size == len(targets), "Mismatch between batch and targets"

        # initialise accumulators
        l_box = torch.tensor(0.0, device=device)
        l_obj_pos = torch.tensor(0.0, device=device)
        l_obj_neg = torch.tensor(0.0, device=device)
        l_cls = torch.tensor(0.0, device=device)
        l_ang = torch.tensor(0.0, device=device)
        l_dfl = torch.tensor(0.0, device=device)
        l_kpt = torch.tensor(0.0, device=device)
        # accumulate IoU penalty across positive samples when enabled
        l_iou = torch.tensor(0.0, device=device)
        eps = 1e-6

        # precompute grids for each level
        grid_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for dm in det_maps:
            _, _, h, w = dm.shape
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
            )
            grid_cache.append((xx.float(), yy.float()))

        # iterate over levels and split predictions
        # For negative loss we accumulate BCE over entire map; for positive we override cells assigned later
        neg_obj_masks: List[torch.Tensor] = []
        neg_cls_masks: List[torch.Tensor] = []
        for level, dm in enumerate(det_maps):
            B, C, H, W = dm.shape
            # channel offsets
            idx = 0
            tx = dm[:, idx:idx + 1]; idx += 1
            ty = dm[:, idx:idx + 1]; idx += 1
            dfl_w = dm[:, idx:idx + self.nbins]; idx += self.nbins
            dfl_h = dm[:, idx:idx + self.nbins]; idx += self.nbins
            ang = dm[:, idx:idx + 1]; idx += 1
            obj = dm[:, idx:idx + 1]; idx += 1
            cls = dm[:, idx:] if C > idx else None
            # for all negatives: objectness and classification targets are zero
            neg_obj_masks.append(obj)
            if cls is not None and cls.shape[1] > 0:
                neg_cls_masks.append(cls)
            else:
                neg_cls_masks.append(torch.zeros_like(obj))

        # now assign ground truth boxes to levels and cells
        # We'll create masks for positive cells and gather predicted values
        for b, tgt in enumerate(targets):
            if tgt is None or tgt.get("boxes") is None or tgt["boxes"].numel() == 0:
                continue
            boxes = tgt["boxes"].to(device)  # (N,5)
            labels = tgt.get("labels")
            if labels is not None:
                labels = labels.to(device)
            kpts = tgt.get("keypoints")
            if kpts is not None:
                kpts = kpts.to(device)
            n_gt = boxes.shape[0]
            for gi in range(n_gt):
                cx_gt, cy_gt, w_gt, h_gt, ang_gt = boxes[gi]
                max_side = float(torch.max(w_gt, h_gt))
                lvl = self._get_level(max_side)
                stride = self.strides[lvl]
                # compute grid indices of the centre on this level
                xi = int(torch.clamp(torch.floor(cx_gt / stride), min=0, max=det_maps[lvl].shape[3] - 1).item())
                yi = int(torch.clamp(torch.floor(cy_gt / stride), min=0, max=det_maps[lvl].shape[2] - 1).item())
                # compute neighbour cells within manhattan distance <= neighbor_range
                for dy in range(-self.neighbor_range, self.neighbor_range + 1):
                    for dx in range(-self.neighbor_range, self.neighbor_range + 1):
                        # when neighbour_cells is disabled, skip all offsets except (0,0)
                        if not self.neighbor_cells and (dx != 0 or dy != 0):
                            continue
                        # apply Manhattan distance limit
                        if abs(dx) + abs(dy) > self.neighbor_range:
                            continue
                        x_idx = xi + dx
                        y_idx = yi + dy
                        if not (0 <= x_idx < det_maps[lvl].shape[3] and 0 <= y_idx < det_maps[lvl].shape[2]):
                            continue
                        # mark this cell as positive; compute predictions and losses
                        dm = det_maps[lvl]
                        B, C, H, W = dm.shape
                        # select predicted values at (b, :, y_idx, x_idx)
                        idx0 = 0
                        txp = dm[b, idx0, y_idx, x_idx]; idx0 += 1
                        typ = dm[b, idx0, y_idx, x_idx]; idx0 += 1
                        dflw = dm[b, idx0:idx0 + self.nbins, y_idx, x_idx]; idx0 += self.nbins
                        dflh = dm[b, idx0:idx0 + self.nbins, y_idx, x_idx]; idx0 += self.nbins
                        angp = dm[b, idx0, y_idx, x_idx]; idx0 += 1
                        objp = dm[b, idx0, y_idx, x_idx]; idx0 += 1
                        clsp = dm[b, idx0:, y_idx, x_idx] if (C > idx0) else None

                        # decode centre (cx, cy) from txp/typ
                        gx, gy = grid_cache[lvl][0][y_idx, x_idx], grid_cache[lvl][1][y_idx, x_idx]
                        cx_pred = (torch.sigmoid(txp) + gx) * stride
                        cy_pred = (torch.sigmoid(typ) + gy) * stride
                        # decode w/h via expected value of DFL
                        pw = (F.softmax(dflw, dim=0) * self.bins.to(device=dflw.device)).sum() * stride
                        ph = (F.softmax(dflh, dim=0) * self.bins.to(device=dflh.device)).sum() * stride
                        # decode angle to [-pi/2, pi/2]
                        ang_pred = (torch.sigmoid(angp) * 2.0 - 1.0) * (math.pi / 2.0)
                        # canonicalise predicted box
                        pw_c, ph_c, ang_c = self._le90(pw.unsqueeze(0), ph.unsqueeze(0), ang_pred.unsqueeze(0))
                        pw_c, ph_c, ang_c = pw_c[0], ph_c[0], ang_c[0]
                        # compute box regression losses (centre and log‑size)
                        l_box = l_box + F.smooth_l1_loss(cx_pred, cx_gt, reduction='mean')
                        l_box = l_box + F.smooth_l1_loss(cy_pred, cy_gt, reduction='mean')
                        l_box = l_box + F.smooth_l1_loss(torch.log(pw_c + eps), torch.log(w_gt + eps), reduction='mean')
                        l_box = l_box + F.smooth_l1_loss(torch.log(ph_c + eps), torch.log(h_gt + eps), reduction='mean')
                        # angle regression
                        l_ang = l_ang + F.smooth_l1_loss(ang_c, ang_gt, reduction='mean')
                        # DFL losses: soft target on bins for w and h
                        # compute gt in stride units
                        w_s = (w_gt / stride).clamp(min=0, max=self.reg_max - 0.001)
                        h_s = (h_gt / stride).clamp(min=0, max=self.reg_max - 0.001)
                        w_bin = torch.floor(w_s)
                        h_bin = torch.floor(h_s)
                        w_rem = (w_s - w_bin).clamp(min=0.0, max=1.0)
                        h_rem = (h_s - h_bin).clamp(min=0.0, max=1.0)
                        # weights for bins: assign to floor and floor+1
                        # create target distribution on the fly
                        # for width
                        if self.lambda_dfl > 0.0:
                            w_bin = int(w_bin.item())
                            h_bin = int(h_bin.item())
                            # clamp indices
                            w_next = min(w_bin + 1, self.reg_max)
                            h_next = min(h_bin + 1, self.reg_max)
                            # compute cross‑entropy manually using log_softmax
                            lw = F.log_softmax(dflw, dim=0)
                            lh = F.log_softmax(dflh, dim=0)
                            l_dfl_w = -((1.0 - w_rem) * lw[w_bin] + (w_rem) * lw[w_next])
                            l_dfl_h = -((1.0 - h_rem) * lh[h_bin] + (h_rem) * lh[h_next])
                            l_dfl = l_dfl + l_dfl_w + l_dfl_h
                        # optional IoU penalty (AABB IoU) on predicted vs ground truth boxes
                        if self.lambda_iou > 0.0:
                            # compute AABB IoU: predicted centre and size (after canonicalisation)
                            px1 = cx_pred - 0.5 * pw_c
                            py1 = cy_pred - 0.5 * ph_c
                            px2 = cx_pred + 0.5 * pw_c
                            py2 = cy_pred + 0.5 * ph_c
                            gx1 = cx_gt - 0.5 * w_gt
                            gy1 = cy_gt - 0.5 * h_gt
                            gx2 = cx_gt + 0.5 * w_gt
                            gy2 = cy_gt + 0.5 * h_gt
                            inter_w = torch.clamp_min(torch.min(px2, gx2) - torch.max(px1, gx1), 0.0)
                            inter_h = torch.clamp_min(torch.min(py2, gy2) - torch.max(py1, gy1), 0.0)
                            inter_area = inter_w * inter_h
                            union_area = pw_c * ph_c + w_gt * h_gt - inter_area + 1e-9
                            iou = inter_area / union_area
                            # accumulate 1 - IoU (lower is better)
                            l_iou = l_iou + (1.0 - iou)
                        # positive objectness
                        l_obj_pos = l_obj_pos + F.binary_cross_entropy_with_logits(objp, torch.tensor(1.0, device=device))
                        # positive classification (multi‑label BCE with single positive class)
                        if self.num_classes > 0:
                            if clsp is not None and clsp.numel() > 0 and labels is not None:
                                target_cls = torch.zeros(self.num_classes, device=device)
                                target_cls[int(labels[gi].item())] = 1.0
                                l_cls = l_cls + F.binary_cross_entropy_with_logits(clsp, target_cls)
                        # keypoint loss
                        if self.use_kpt and kpts is not None and kpts.shape[0] > gi:
                            # decode keypoint from keypoint map at same cell
                            km = kpt_maps[lvl]
                            kpx = km[b, 0, y_idx, x_idx]
                            kpy = km[b, 1, y_idx, x_idx]
                            kx_pred = (torch.sigmoid(kpx) + gx) * stride
                            ky_pred = (torch.sigmoid(kpy) + gy) * stride
                            kx_gt, ky_gt = kpts[gi]
                            l_kpt = l_kpt + F.smooth_l1_loss(kx_pred, kx_gt, reduction='mean')
                            l_kpt = l_kpt + F.smooth_l1_loss(ky_pred, ky_gt, reduction='mean')
                        # mark this cell as processed for negative loss by setting obj to None
                        # to avoid counting it as negative later we will zero out its logits in neg losses
                        neg_obj_masks[lvl][b, 0, y_idx, x_idx] = torch.nan  # sentinel
                        if neg_cls_masks[lvl].numel() > 0:
                            neg_cls_masks[lvl][b, :, y_idx, x_idx] = torch.nan

        # compute negative objectness and classification losses for all unassigned cells
        for lvl, dm in enumerate(det_maps):
            obj = neg_obj_masks[lvl]
            cls = neg_cls_masks[lvl]
            # ignore NaNs (assigned positives)
            valid_mask = ~torch.isnan(obj)
            if valid_mask.any():
                l_obj_neg = l_obj_neg + F.binary_cross_entropy_with_logits(obj[valid_mask], torch.zeros_like(obj[valid_mask]))
            if self.num_classes > 0 and cls.numel() > 0:
                valid_mask_cls = ~torch.isnan(cls)
                if valid_mask_cls.any():
                    l_cls = l_cls + F.binary_cross_entropy_with_logits(cls[valid_mask_cls], torch.zeros_like(cls[valid_mask_cls]))

        # normalise losses by number of positives to stabilise training
        pos_count = max(1.0, l_obj_pos.detach().item())  # use objectness pos count as proxy
        # sum and weight components, including optional IoU penalty
        loss = (
            self.lambda_box * l_box
            + self.lambda_ang * l_ang
            + self.lambda_dfl * l_dfl
            + self.lambda_obj * (l_obj_pos + l_obj_neg)
            + self.lambda_cls * l_cls
        )
        if self.lambda_iou > 0.0:
            loss = loss + self.lambda_iou * l_iou
        if self.use_kpt:
            loss = loss + self.lambda_kpt * l_kpt
        # divide by pos_count to roughly normalise magnitude
        loss = loss / pos_count
        loss_dict = {
            "loss_box": l_box / pos_count,
            "loss_ang": l_ang / pos_count,
            "loss_dfl": l_dfl / pos_count,
            "loss_obj_pos": l_obj_pos / pos_count,
            "loss_obj_neg": l_obj_neg / pos_count,
            "loss_cls": l_cls / pos_count,
            "loss_kpt": (l_kpt / pos_count) if self.use_kpt else torch.tensor(0.0, device=device),
            "loss_iou": (l_iou / pos_count) if self.lambda_iou > 0.0 else torch.tensor(0.0, device=device),
        }
        return loss, loss_dict