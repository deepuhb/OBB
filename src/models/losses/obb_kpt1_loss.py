# src/models/losses/obb_kpt1_loss.py
import math
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.box_ops import bbox_iou_xyxy


class OBBKpt1Criterion(nn.Module):
    """
    YOLO11-style criterion for:
        - Oriented box (angle via sin/cos) but trained on AABB (xyxy) for stability
        - Single keypoint per object, parameterized as (u, v) in [0,1] within AABB
        - Objectness with warm-up (hard=1) then soft targets (IoU)
        - Optional 4-neighbor assignment around the center cell on each FPN level

    Inputs to forward():
        det_maps: List[Tensor]  # per-level det logits [B, 7(+nc), H, W]
                                # channels: tx, ty, tw, th, sin, cos, obj   (+ cls if nc>1)
        kpt_maps: List[Tensor]  # per-level kpt logits [B, 3, H, W] -> (u, v, kconf_logit)
        batch:    Dict with:
            image:  Tensor [B, 3, Himg, Wimg]
            boxes:  list[Tensor (Ni, 4)]  GT AABBs (xyxy) in pixels
            kpts:   list[Tensor (Ni, 2)]  GT keypoints (x, y) in pixels
            angles: list[Tensor (Ni,)]    GT angles (radians)
            labels: list[Tensor (Ni,)]    GT classes (unused if 1 class)
        epoch: int
        soft_warmup_epochs: int

    Returns:
        total_loss: Tensor (scalar)
        logs: dict of python floats
    """
    def __init__(
        self,
        strides: Tuple[int, int, int] = (4, 8, 16),
        num_classes: int = 1,
        lambda_box: float = 7.5,
        lambda_obj: float = 3.0,
        lambda_ang: float = 1.0,
        lambda_kpt: float = 2.0,
        lambda_kc: float = 0.5,
        pos_obj_weight: float = 1.0,
        soft_obj_warmup_epochs: int = 2,
        assign_4_neighbors: bool = True,
        box_loss_type: str = "iou"  # "iou" or "giou"
    ):
        super().__init__()
        self.strides = tuple(strides)
        self.num_classes = num_classes

        self.lb_box = float(lambda_box)
        self.lb_obj = float(lambda_obj)
        self.lb_ang = float(lambda_ang)
        self.lb_kpt = float(lambda_kpt)
        self.lb_kc  = float(lambda_kc)

        self.pos_obj_weight = float(pos_obj_weight)
        self.soft_obj_warmup_epochs = int(soft_obj_warmup_epochs)
        self.assign_4_neighbors = bool(assign_4_neighbors)
        self.box_loss_type = str(box_loss_type).lower()

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.eps = 1e-9

    # ------------------------- small helpers -------------------------

    @staticmethod
    def _grid(h: int, w: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij"
        )
        return xs, ys

    def _decode_level(self, pred: torch.Tensor, stride: int):
        """
        pred: [B, C, H, W]  channels = tx, ty, tw, th, sin, cos, obj, ...
        returns:
            boxes_dec: [B, H, W, 4]    (xyxy)
            obj_log:   [B, H, W, 1]    (logits)
            sin_map:   [B, H, W, 1]
            cos_map:   [B, H, W, 1]
        """
        B, C, H, W = pred.shape
        gx, gy = self._grid(H, W, pred.device)

        tx = pred[:, 0].sigmoid()
        ty = pred[:, 1].sigmoid()
        tw = pred[:, 2].exp().clamp(1e-3, 1e6)
        th = pred[:, 3].exp().clamp(1e-3, 1e6)
        sin = pred[:, 4:5]
        cos = pred[:, 5:6]
        obj = pred[:, 6:7]  # logits

        cx = (tx + gx) * stride
        cy = (ty + gy) * stride
        w  = tw * stride
        h  = th * stride

        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [B,H,W,4]
        return boxes, obj.permute(0, 2, 3, 1), sin.permute(0, 2, 3, 1), cos.permute(0, 2, 3, 1)

    @staticmethod
    def _boxes_center_size(xyxy: torch.Tensor):
        """xyxy -> (cx, cy, w, h)"""
        cx = (xyxy[..., 0] + xyxy[..., 2]) * 0.5
        cy = (xyxy[..., 1] + xyxy[..., 3]) * 0.5
        w  = (xyxy[..., 2] - xyxy[..., 0]).clamp_min(1.0)
        h  = (xyxy[..., 3] - xyxy[..., 1]).clamp_min(1.0)
        return cx, cy, w, h

    @staticmethod
    def _giou_xyxy(a: torch.Tensor, b: torch.Tensor):
        """
        Generalized IoU for xyxy pairs.
        a: [N,4], b: [N,4]
        """
        # IoU
        tl = torch.maximum(a[..., :2], b[..., :2])
        br = torch.minimum(a[..., 2:], b[..., 2:])
        wh = (br - tl).clamp(min=0.0)
        inter = wh[..., 0] * wh[..., 1]

        area_a = (a[..., 2] - a[..., 0]).clamp(min=0.0) * (a[..., 3] - a[..., 1]).clamp(min=0.0)
        area_b = (b[..., 2] - b[..., 0]).clamp(min=0.0) * (b[..., 3] - b[..., 1]).clamp(min=0.0)
        union = (area_a + area_b - inter).clamp(min=1e-9)
        iou = inter / union

        # smallest enclosing box
        c_tl = torch.minimum(a[..., :2], b[..., :2])
        c_br = torch.maximum(a[..., 2:], b[..., 2:])
        c_wh = (c_br - c_tl).clamp(min=0.0)
        c_area = c_wh[..., 0] * c_wh[..., 1] + 1e-9

        giou = iou - (c_area - union) / c_area
        return giou

    # ------------------------- target building -------------------------

    def build_targets(
        self,
        batch: Dict[str, List[torch.Tensor]],
        feat_shapes: List[Tuple[int, int]],
    ):
        """
        Produces per-level dense targets:
            obj_t: [B,H,W,1] in {0,1}
            box_t: [B,H,W,4] xyxy
            ang_t: [B,H,W,2] (sinθ, cosθ)
            uv_t : [B,H,W,2] (u,v) in [0,1] relative to AABB
            vis_t: [B,H,W,1] visibility (1 for our single KP)
        """
        B = len(batch["boxes"])
        dev = batch["image"].device
        outs = []

        for (H, W), s in zip(feat_shapes, self.strides):
            obj_t = torch.zeros((B, H, W, 1), device=dev)
            box_t = torch.zeros((B, H, W, 4), device=dev)
            ang_t = torch.zeros((B, H, W, 2), device=dev)  # sin, cos
            uv_t  = torch.zeros((B, H, W, 2), device=dev)
            vis_t = torch.zeros((B, H, W, 1), device=dev)

            for b in range(B):
                gtb = batch["boxes"][b].to(dev)       # (N,4)
                gtk = batch["kpts"][b].to(dev)        # (N,2)
                ang = batch["angles"][b].to(dev) if b < len(batch["angles"]) else torch.zeros((gtb.shape[0],), device=dev)

                if gtb.numel() == 0:
                    continue

                cx, cy, w, h = self._boxes_center_size(gtb)
                # cell indices
                gi = (cy / s).long().clamp(0, H - 1)
                gj = (cx / s).long().clamp(0, W - 1)

                for n in range(gtb.shape[0]):
                    cells = [(gi[n].item(), gj[n].item())]
                    if self.assign_4_neighbors:
                        i0, j0 = cells[0]
                        for ii, jj in ((i0 - 1, j0), (i0 + 1, j0), (i0, j0 - 1), (i0, j0 + 1)):
                            if 0 <= ii < H and 0 <= jj < W:
                                cells.append((ii, jj))
                    # de-duplicate
                    cells = list(dict.fromkeys(cells))

                    for (i, j) in cells:
                        obj_t[b, i, j, 0] = 1.0
                        box_t[b, i, j, :] = gtb[n]

                        # angle target as (sin, cos)
                        ang_t[b, i, j, 0] = torch.sin(ang[n])
                        ang_t[b, i, j, 1] = torch.cos(ang[n])

                        # keypoint (u,v) in [0,1] within AABB
                        kx, ky = gtk[n]
                        u = ((kx - cx[n]) / w[n] + 0.5).clamp(0.0, 1.0)
                        v = ((ky - cy[n]) / h[n] + 0.5).clamp(0.0, 1.0)
                        uv_t[b, i, j, 0] = u
                        uv_t[b, i, j, 1] = v
                        vis_t[b, i, j, 0] = 1.0

            outs.append((obj_t, box_t, ang_t, uv_t, vis_t))
        return outs

    # ------------------------- forward (loss) -------------------------

    def forward(
        self,
        det_maps: List[torch.Tensor],
        kpt_maps: List[torch.Tensor],
        batch: Dict[str, torch.Tensor],
        epoch: int = 0,
        soft_warmup_epochs: int = 2,
    ):
        feat_shapes = [(m.shape[2], m.shape[3]) for m in det_maps]
        targets = self.build_targets(batch, feat_shapes)

        losses = dict(obj=0.0, box=0.0, ang=0.0, kpt=0.0, kc=0.0)
        num_pos_total = 0
        iou_mean_running = None

        for lvl, (pred_det, pred_kp, s) in enumerate(zip(det_maps, kpt_maps, self.strides)):
            # decode current level
            boxes_dec, obj_log, sin_map, cos_map = self._decode_level(pred_det, s)
            obj_t, box_t, ang_t, uv_t, vis_t = targets[lvl]

            # positive mask
            pos = obj_t[..., 0] > 0.5
            num_pos_total += int(pos.sum().item())

            # -------- objectness (warm-up -> soft IoU targets) --------
            if epoch < soft_warmup_epochs:
                target_map = obj_t[..., 0]  # hard 0/1
            else:
                if pos.any():
                    iou_here = bbox_iou_xyxy(boxes_dec[pos], box_t[pos]).clamp(0.0, 1.0)
                else:
                    iou_here = torch.zeros((0,), device=obj_t.device)
                target_map = torch.zeros_like(obj_t[..., 0])
                if pos.any():
                    target_map[pos] = iou_here.detach()

            # weighting: optionally upweight positive locations
            w = torch.ones_like(target_map)
            if self.pos_obj_weight != 1.0:
                w[pos] = self.pos_obj_weight

            bce = F.binary_cross_entropy_with_logits(
                obj_log.squeeze(-1), target_map, reduction="none"
            )
            losses["obj"] += (bce * w).sum() / w.sum().clamp_min(1.0)

            # -------- box (1 - IoU) or (1 - GIoU) on positives --------
            if pos.any():
                iou = bbox_iou_xyxy(boxes_dec[pos], box_t[pos]).clamp(0.0, 1.0)
                if self.box_loss_type == "giou":
                    giou = self._giou_xyxy(boxes_dec[pos], box_t[pos]).clamp(-1.0, 1.0)
                    losses["box"] += (1.0 - giou).mean()
                    iou_for_log = iou
                else:
                    losses["box"] += (1.0 - iou).mean()
                    iou_for_log = iou

                # keep a running mean IoU for logging
                iou_mean_running = (
                    iou_for_log.mean()
                    if iou_mean_running is None
                    else 0.5 * iou_mean_running + 0.5 * iou_for_log.mean()
                )

                # -------- angle loss on (sin, cos) --------
                # Normalize predicted (sin,cos) to unit circle before computing MSE,
                # which makes it equivalent to 1 - cos(Δθ) up to a scale.
                s_pred = sin_map[pos][..., 0]
                c_pred = cos_map[pos][..., 0]
                vec = torch.stack([s_pred, c_pred], dim=-1)
                vec = vec / (vec.norm(dim=-1, keepdim=True) + self.eps)

                s_true = ang_t[pos][..., 0]
                c_true = ang_t[pos][..., 1]
                tgt = torch.stack([s_true, c_true], dim=-1)  # already unit-length

                losses["ang"] += F.mse_loss(vec, tgt)

                # -------- keypoint loss --------
                # predicted (u,v) must be in [0,1] -> apply sigmoid
                kp = pred_kp.permute(0, 2, 3, 1)  # [B,H,W,3]
                uv_pred = kp[pos][..., :2].sigmoid()
                uv_true = uv_t[pos]
                vis = vis_t[pos][..., 0]  # {0,1}

                if vis.sum() > 0:
                    l1 = F.smooth_l1_loss(uv_pred, uv_true, reduction="none").sum(dim=-1)
                    losses["kpt"] += (l1 * vis).sum() / vis.sum().clamp_min(1.0)

                # keypoint confidence (logits) vs visibility
                if pred_kp.shape[1] >= 3:
                    kc_log = kp[pos][..., 2]  # logits
                    losses["kc"] += F.binary_cross_entropy_with_logits(kc_log, vis)

        # total weighted loss
        total = (
            self.lb_obj * losses["obj"]
            + self.lb_box * losses["box"]
            + self.lb_ang * losses["ang"]
            + self.lb_kpt * losses["kpt"]
            + self.lb_kc  * losses["kc"]
        )

        # convert logs to python floats (no .item() on plain floats)
        def _f(x):  # safe float
            return float(x if isinstance(x, (int, float)) else x.detach().item())

        logs = dict(
            loss=_f(total),
            loss_obj=_f(losses["obj"]),
            loss_box=_f(losses["box"]),
            loss_ang=_f(losses["ang"]),
            loss_kpt=_f(losses["kpt"]),
            loss_kc=_f(losses["kc"]),
            num_pos=float(num_pos_total),
            mean_iou=_f(iou_mean_running if iou_mean_running is not None else 0.0),
        )
        return total, logs
