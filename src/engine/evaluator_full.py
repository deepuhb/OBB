"""
Evaluator for oriented bounding boxes and keypoints.

This module provides an ``EvaluatorFull`` class that can be used in place of
the original axis‑aligned evaluator.  It decodes model outputs into rotated
bounding boxes (defined by centre x/y, width, height and angle) and keypoints,
performs a greedy non‑maximum suppression based on oriented IoU, and
computes average precision (mAP) and PCK metrics.

Key features:
  • Oriented IoU using shapely polygons.
  • Greedy rotated NMS to filter duplicate detections.
  • PCK computed both on matched detections (PCK) and across all detections
    without requiring a box match (PCK_any) at a configurable tau.
  • Optional recall statistics at different IoU thresholds.

The API mirrors the original ``Evaluator``: construct the evaluator with
thresholds and then call ``evaluate(model, loader, device)`` to compute
metrics.  The return value is a dict containing ``map50``, ``pck@tau``,
``pck_any@tau``, ``tps`` (true positives), ``pred_per_img``, ``images`` and
``best_iou``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import shapely.geometry
import shapely.affinity


def _obb_to_polygon(cx: float, cy: float, w: float, h: float, angle: float) -> List[Tuple[float, float]]:
    """Return the four corner coordinates of a rotated rectangle.

    The rectangle is defined by its centre ``(cx, cy)``, width ``w``,
    height ``h`` and rotation angle ``angle`` in radians.  The corners are
    returned as a list of (x, y) tuples in clockwise order starting from
    the top‑left corner.
    """
    # axis‑aligned corners around the origin
    hw, hh = w * 0.5, h * 0.5
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    pts: List[Tuple[float, float]] = []
    for x, y in corners:
        xr = x * cos_a - y * sin_a
        yr = x * sin_a + y * cos_a
        pts.append((xr + cx, yr + cy))
    return pts


def _polygon_iou(poly1: List[Tuple[float, float]], poly2: List[Tuple[float, float]]) -> float:
    """Compute the IoU between two polygons.

    Each polygon is represented as a list of (x, y) coordinates.  The
    shapely library is used to compute intersection and union areas.  If the
    union is zero the function returns 0.0.
    """
    try:
        p1 = shapely.geometry.Polygon(poly1)
        p2 = shapely.geometry.Polygon(poly2)
    except Exception:
        return 0.0
    if not p1.is_valid or not p2.is_valid:
        return 0.0
    inter = p1.intersection(p2).area
    if inter <= 0.0:
        return 0.0
    union = p1.union(p2).area
    return inter / union if union > 0 else 0.0


#
# Detection decoding for top-down OBB head
#
@torch.no_grad()
def _decode_det_maps(
    det_maps: List[torch.Tensor],
    strides: Tuple[int, ...],
    *,
    score_thresh: float = 0.3,
    nms_iou: float = 0.5,
    max_det: int = 100,
) -> List[Dict[str, Any]]:
    """Decode detection head feature maps into rotated bounding boxes.

    Args:
        det_maps: list of FPN level tensors, each of shape (B,7+nc,H,W) with channels
                  [tx,ty,tw,th,sin,cos,obj,cls...]. Only single-class is supported.
        strides: tuple of strides for each FPN level.
        score_thresh: pre-threshold on objectness score.
        nms_iou: IoU threshold for greedy rotated NMS.
        max_det: maximum number of detections to keep per image after NMS.

    Returns:
        A list of length B. Each element is a dict with keys:
          ``obb`` – tensor of shape (Ni,5) [cx,cy,w,h,deg],
          ``scores`` – tensor of length Ni,
          ``labels`` – tensor of length Ni (zeros for single class),
          ``polygons`` – list of polygons [(x,y)*4] for IoU computations.

    Notes:
        This decoder matches the YOLO-style OBB head used in this project (sin,cos activated with tanh; width/height exponential). It does not handle multi-class.
    """
    device = det_maps[0].device
    batch_size = det_maps[0].shape[0]
    outputs: List[Dict[str, Any]] = []
    for b in range(batch_size):
        cx_all: List[float] = []
        cy_all: List[float] = []
        w_all:  List[float] = []
        h_all:  List[float] = []
        ang_all: List[float] = []
        scores_all: List[float] = []
        for dm, s in zip(det_maps, strides):
            logits = dm[b]  # (C,H,W)
            H, W = logits.shape[1], logits.shape[2]
            # decode regression channels
            tx = logits[0].sigmoid(); ty = logits[1].sigmoid()
            tw = logits[2].exp();     th = logits[3].exp()
            # orientation: sin, cos via tanh
            si = logits[4].tanh();    co = logits[5].tanh()
            obj= logits[6].sigmoid()
            sc = obj  # single-class
            # grid
            ys, xs = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij"
            )
            cx = (tx + xs) * s
            cy = (ty + ys) * s
            w  = tw.clamp_min(1e-6) * s
            h  = th.clamp_min(1e-6) * s
            ang = torch.atan2(si, co)
            keep = sc > float(score_thresh)
            if keep.sum() == 0:
                continue
            cx_keep = cx[keep]; cy_keep = cy[keep]
            w_keep  = w[keep];  h_keep  = h[keep]
            ang_keep= ang[keep]
            sc_keep = sc[keep]
            cx_all.extend(cx_keep.tolist())
            cy_all.extend(cy_keep.tolist())
            w_all.extend(w_keep.tolist())
            h_all.extend(h_keep.tolist())
            ang_all.extend(ang_keep.tolist())
            scores_all.extend(sc_keep.tolist())
        if not scores_all:
            outputs.append({
                "obb": torch.zeros((0,5), device=device),
                "scores": torch.zeros((0,), device=device),
                "labels": torch.zeros((0,), dtype=torch.long, device=device),
                "polygons": []
            })
            continue
        cx_t = torch.tensor(cx_all, device=device)
        cy_t = torch.tensor(cy_all, device=device)
        w_t  = torch.tensor(w_all, device=device)
        h_t  = torch.tensor(h_all, device=device)
        ang_t= torch.tensor(ang_all, device=device)
        scores_t = torch.tensor(scores_all, device=device)
        labels_t = torch.zeros_like(scores_t, dtype=torch.long)
        ang_deg = ang_t * (180.0 / math.pi)
        polys: List[List[Tuple[float, float]]] = []
        for cx_val, cy_val, w_val, h_val, ang_rad in zip(cx_t.tolist(), cy_t.tolist(), w_t.tolist(), h_t.tolist(), ang_t.tolist()):
            polys.append(_obb_to_polygon(cx_val, cy_val, w_val, h_val, ang_rad))
        # greedy NMS using shapely polygons
        order = torch.argsort(scores_t, descending=True).tolist()
        keep_indices: List[int] = []
        for idx in order:
            keep_it = True
            for kidx in keep_indices:
                if _polygon_iou(polys[idx], polys[kidx]) > nms_iou:
                    keep_it = False
                    break
            if keep_it:
                keep_indices.append(idx)
            if len(keep_indices) >= max_det:
                break
        keep_t = torch.tensor(keep_indices, device=device, dtype=torch.long)
        outputs.append({
            "obb": torch.stack([cx_t, cy_t, w_t, h_t, ang_deg], dim=1)[keep_t],
            "scores": scores_t[keep_t],
            "labels": labels_t[keep_t],
            "polygons": [polys[i] for i in keep_indices]
        })
    return outputs


@torch.no_grad()
def decode_rotated(
    det_maps: List[torch.Tensor],
    kpt_maps: List[torch.Tensor],
    *,
    strides: Tuple[int, int, int] = (4, 8, 16),
    score_thresh: float = 0.3,
    iou_thresh: float = 0.5,
    max_det: int = 100,
) -> List[Dict[str, torch.Tensor]]:
    """Decode model predictions into rotated bounding boxes and keypoints.

    The model outputs dense predictions on feature maps of different strides.
    This function converts those predictions into per‑image detections, one
    list per batch element.  Each detection dict contains:

      ``boxes``  – tensor of shape (N, 5) with columns (cx, cy, w, h, angle).
      ``polygons`` – list of lists of 4 (x, y) tuples describing the rotated box.
      ``scores`` – tensor of length N with objectness scores.
      ``labels`` – tensor of zeros (single class).
      ``kpts``   – tensor of shape (N, 2) with absolute keypoint positions.

    Non‑maximum suppression is performed greedily on the rotated boxes using
    IoU computed on the polygons.  The ``max_det`` parameter limits the
    number of detections per image.
    """
    device = det_maps[0].device
    batch_size = det_maps[0].shape[0]
    outputs: List[Dict[str, torch.Tensor]] = []
    for b in range(batch_size):
        # accumulate predictions from all FPN levels
        cx_all: List[float] = []
        cy_all: List[float] = []
        w_all:  List[float] = []
        h_all:  List[float] = []
        ang_all: List[float] = []
        scores_all: List[float] = []
        kpts_all_x: List[float] = []
        kpts_all_y: List[float] = []
        for dm, km, stride in zip(det_maps, kpt_maps, strides):
            logits = dm[b]
            H, W = logits.shape[1], logits.shape[2]
            # predicted parameters
            tx = logits[0].sigmoid()
            ty = logits[1].sigmoid()
            tw = logits[2].exp()
            th = logits[3].exp()
            sin = logits[4]
            cos = logits[5]
            obj = logits[6].sigmoid()
            # grid
            ys, xs = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij",
            )
            # convert to absolute scale
            cx = (tx + xs) * stride
            cy = (ty + ys) * stride
            w  = tw.clamp_min(1e-3) * stride
            h  = th.clamp_min(1e-3) * stride
            ang = torch.atan2(sin, cos)
            # mask by objectness
            keep = obj > score_thresh
            if keep.sum() == 0:
                continue
            # gather values
            cx_keep = cx[keep]
            cy_keep = cy[keep]
            w_keep  = w[keep]
            h_keep  = h[keep]
            ang_keep = ang[keep]
            scores_keep = obj[keep]
            # keypoints (relative offsets within bounding box)
            uv = km[b].permute(1, 2, 0).sigmoid()  # (H, W, 3)
            u = uv[..., 0][keep]
            v = uv[..., 1][keep]
            # relative offsets scaled by w/h
            kx_rel = (u - 0.5) * w_keep
            ky_rel = (v - 0.5) * h_keep
            # rotate relative offset and translate to absolute
            cos_a = ang_keep.cos()
            sin_a = ang_keep.sin()
            kx = kx_rel * cos_a - ky_rel * sin_a + cx_keep
            ky = kx_rel * sin_a + ky_rel * cos_a + cy_keep
            # extend lists
            cx_all.extend(cx_keep.tolist())
            cy_all.extend(cy_keep.tolist())
            w_all.extend(w_keep.tolist())
            h_all.extend(h_keep.tolist())
            ang_all.extend(ang_keep.tolist())
            scores_all.extend(scores_keep.tolist())
            kpts_all_x.extend(kx.tolist())
            kpts_all_y.extend(ky.tolist())
        # build detections per image
        if not scores_all:
            outputs.append({
                "boxes": torch.zeros((0, 5), device=device),
                "scores": torch.zeros((0,), device=device),
                "labels": torch.zeros((0,), dtype=torch.long, device=device),
                "kpts": torch.zeros((0, 2), device=device),
                "polygons": [],
            })
            continue
        # convert to tensors
        cx_t = torch.tensor(cx_all, device=device)
        cy_t = torch.tensor(cy_all, device=device)
        w_t  = torch.tensor(w_all, device=device)
        h_t  = torch.tensor(h_all, device=device)
        ang_t = torch.tensor(ang_all, device=device)
        scores_t = torch.tensor(scores_all, device=device)
        labels_t = torch.zeros_like(scores_t, dtype=torch.long)
        kpts_t = torch.stack([torch.tensor(kpts_all_x, device=device), torch.tensor(kpts_all_y, device=device)], dim=1)
        # polygons list for NMS and IoU
        polys: List[List[Tuple[float, float]]] = []
        for cx_val, cy_val, w_val, h_val, ang_val in zip(cx_t.tolist(), cy_t.tolist(), w_t.tolist(), h_t.tolist(), ang_t.tolist()):
            polys.append(_obb_to_polygon(cx_val, cy_val, w_val, h_val, ang_val))
        # greedy rotated NMS
        order = torch.argsort(scores_t, descending=True).tolist()
        keep: List[int] = []
        for idx in order:
            keep_it = True
            for kidx in keep:
                if _polygon_iou(polys[idx], polys[kidx]) > iou_thresh:
                    keep_it = False
                    break
            if keep_it:
                keep.append(idx)
            if len(keep) >= max_det:
                break
        keep_t = torch.tensor(keep, device=device, dtype=torch.long)
        outputs.append({
            "boxes": torch.stack([cx_t, cy_t, w_t, h_t, ang_t], dim=1)[keep_t],
            "scores": scores_t[keep_t],
            "labels": labels_t[keep_t],
            "kpts": kpts_t[keep_t],
            "polygons": [polys[i] for i in keep],
        })
    return outputs


class EvaluatorFull:
    """Evaluator for rotated bounding boxes and keypoints.

    This class mirrors the original ``Evaluator`` but operates on rotated
    bounding boxes.  It computes AP (mAP@IoU threshold), PCK on matched
    detections and PCK_any on any prediction, plus recall statistics at
    optional IoU thresholds.
    """

    def __init__(self, cfg: Dict | None = None, debug: bool = False):
        cfg = cfg or {}
        ev = cfg.get("eval", {})
        self.score_thresh = float(ev.get("score_thresh", 0.3))
        self.nms_iou     = float(ev.get("nms_iou", 0.5))
        self.iou_thr     = float(ev.get("iou_thr", 0.5))
        self.pck_tau     = float(ev.get("pck_tau", 0.05))
        self.max_det     = int(ev.get("max_det", 100))
        self.strides     = tuple(cfg.get("model", {}).get("strides", (4, 8, 16)))
        self.debug       = bool(debug)

        # parse top-down config for keypoint cropping
        td = cfg.get("topdown", {})
        # crop size (in pixels) used for the keypoint head
        self.crop_size = int(td.get("crop_size", 64))
        # stride of the first feature map (e.g. 8 for P3) to convert from feature to image coords
        self.feat_down = int(self.strides[0]) if self.strides else 1
        # number of ROIs to sample per image for keypoint inference
        self.kpt_topk  = int(td.get("kpt_topk", 128))
        # chunk size for ROI pooling to avoid OOM
        self.roi_chunk = int(td.get("roi_chunk", 128))
        # score threshold for selecting ROIs in kpt_from_obbs
        self.score_thresh_kpt = float(td.get("score_thresh_kpt", 0.0))

        # thresholds for recall calculation
        self.recall_thresholds = [0.1, 0.3, 0.5]

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        *,
        device: str = "cpu",
        max_images: int | None = None,
    ) -> Dict[str, float]:
        """Evaluate the model on a dataloader.

        Args:
            model: The PyTorch model to evaluate.
            loader: DataLoader providing batches of samples.
            device: The device on which evaluation should run (``"cpu"`` or ``"cuda"``).
            max_images: Optional maximum number of images to process.  If
                ``None``, all images in the loader are processed.  When set
                to an integer N, at most N images are evaluated and the
                results are computed from this subset.

        Returns:
            A dictionary with metrics: ``map50``, ``pck@<tau>``,
            ``pck_any@<tau>``, ``tps`` (true positives), ``pred_per_img``
            (average number of predictions per image), ``images`` (number
            of images evaluated), ``best_iou`` (best IoU encountered) and
            recall values at the configured thresholds.
        """
        model.eval()
        all_scores: List[float] = []
        all_matches: List[int] = []
        total_gt = 0
        pck_hits = 0
        pck_total = 0
        pck_any_hits = 0
        pck_any_total = 0
        total_preds = 0
        total_images = 0
        # recall counts at multiple thresholds
        recall_matches = {thr: 0 for thr in self.recall_thresholds}
        recall_total = 0
        best_iou_global = 0.0
        # counter for processed images when max_images is set
        images_processed = 0
        for batch in loader:
            # if a maximum number of images is specified and exceeded, stop
            if max_images is not None and images_processed >= max_images:
                break
            batch_size = len(batch["image"])
            imgs = batch["image"].to(device)
            outs = model(imgs)
            # decode detection maps into rotated boxes and polygons (single-class)
            preds = _decode_det_maps(
                outs["det"],
                strides=self.strides,
                score_thresh=self.score_thresh,
                nms_iou=self.nms_iou,
                max_det=self.max_det,
            )

            # obtain predicted keypoints using top-down keypoint head, if available
            # preds is a list of length batch_size, each with "obb" and "scores"
            if hasattr(model, "kpt_from_obbs"):
                obb_list = [p["obb"] for p in preds]
                scores_list = [p["scores"] for p in preds]
                # call the model's top-down keypoint inference
                uv_all, metas = model.kpt_from_obbs(
                    outs.get("feats", []), obb_list,
                    scores_list=scores_list,
                    topk=self.kpt_topk,
                    chunk=self.roi_chunk,
                    score_thresh=self.score_thresh_kpt,
                )
                # build predicted keypoints per image (in image coords)
                pred_kpts_img = [[] for _ in range(batch_size)]
                for j, meta in enumerate(metas):
                    bix = int(meta["bix"])
                    uv = uv_all[j]
                    M = meta["M"]
                    # uv in [0,1]; scale to crop pixels
                    uS = uv[0] * self.crop_size
                    vS = uv[1] * self.crop_size
                    # feature coords
                    x_feat = M[0, 0] * uS + M[0, 1] * vS + M[0, 2]
                    y_feat = M[1, 0] * uS + M[1, 1] * vS + M[1, 2]
                    # image coords
                    x_img = x_feat * self.feat_down
                    y_img = y_feat * self.feat_down
                    pred_kpts_img[bix].append(torch.stack([x_img, y_img], dim=0))
                # convert lists to tensors
                pred_kpts_img = [
                    (torch.stack(lst, dim=0) if len(lst) else torch.zeros((0, 2), device=device))
                    for lst in pred_kpts_img
                ]
            else:
                # fallback: no keypoint head; use empty tensors
                pred_kpts_img = [torch.zeros((0, 2), device=device) for _ in range(batch_size)]
            # iterate over each image in the batch
            for i in range(batch_size):
                # stop if we've hit the image limit
                if max_images is not None and images_processed >= max_images:
                    break
                total_images += 1
                images_processed += 1
                # ground truth for this sample
                gt_quads = batch["quads"][i]
                gt_boxes = batch["boxes"][i]
                gt_kpts = batch["kpts"][i]
                n_gt = gt_boxes.shape[0]
                total_gt += int(n_gt)
                # convert ground truth quadrilaterals to polygons and track widths/heights
                gt_polygons: List[List[Tuple[float, float]]] = []
                gt_ws: List[float] = []
                gt_hs: List[float] = []
                for j in range(n_gt):
                    quad = gt_quads[j]
                    poly = [(float(quad[k, 0]), float(quad[k, 1])) for k in range(4)]
                    gt_polygons.append(poly)
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    w = max(xs) - min(xs)
                    h = max(ys) - min(ys)
                    gt_ws.append(w)
                    gt_hs.append(h)
                # predicted detections for this image
                pr = preds[i]
                polys = pr["polygons"]
                scores = pr["scores"]
                # predicted keypoints list for this image (in same order as detections)
                kpts_pred = pred_kpts_img[i]
                n_pred = len(polys)
                total_preds += n_pred
                if self.debug:
                    msg = f"[eval_full] img#{i}: GT={n_gt}, pred={n_pred}"
                    if n_pred:
                        msg += f", score[min/mean/max]=[{scores.min().item():.3f}/{scores.mean().item():.3f}/{scores.max().item():.3f}]"
                    print(msg)
                # matches for AP computation
                taken = np.zeros(n_gt, dtype=bool)
                # compute IoU matrix between predictions and gt polygons
                ious = np.zeros((n_pred, n_gt), dtype=float)
                for pidx in range(n_pred):
                    for gidx in range(n_gt):
                        ious[pidx, gidx] = _polygon_iou(polys[pidx], gt_polygons[gidx])
                        # track global best IoU
                        if ious[pidx, gidx] > best_iou_global:
                            best_iou_global = ious[pidx, gidx]
                # matches for recall thresholds
                recall_total += n_gt
                # list to indicate if gt matched at each recall threshold
                recall_taken = {thr: np.zeros(n_gt, dtype=bool) for thr in self.recall_thresholds}
                # sort predictions by descending score
                order = np.argsort(-scores.cpu().numpy())
                for pidx in order:
                    all_scores.append(float(scores[pidx]))
                    # find best matching gt for this prediction
                    if n_gt > 0:
                        best_idx = int(np.argmax(ious[pidx]))
                        best_iou = ious[pidx, best_idx]
                        if best_iou >= self.iou_thr and not taken[best_idx]:
                            taken[best_idx] = True
                            all_matches.append(1)
                            # PCK on matched pair (use keypoint if available)
                            if pidx < kpts_pred.shape[0]:
                                thr = self.pck_tau * max(gt_ws[best_idx], gt_hs[best_idx])
                                d = float(np.linalg.norm((kpts_pred[pidx].cpu().numpy() - gt_kpts[best_idx].numpy())))
                                pck_hits += int(d <= thr)
                                pck_total += 1
                        else:
                            all_matches.append(0)
                    else:
                        all_matches.append(0)
                    # update recall matches for any thresholds
                    for thr in self.recall_thresholds:
                        if n_gt > 0:
                            best_idx_thr = int(np.argmax(ious[pidx]))
                            if ious[pidx, best_idx_thr] >= thr and not recall_taken[thr][best_idx_thr]:
                                recall_matches[thr] += 1
                                recall_taken[thr][best_idx_thr] = True
                # PCK_any: for each gt keypoint, see if any predicted keypoint is within thr
                for gidx in range(n_gt):
                    thr = self.pck_tau * max(gt_ws[gidx], gt_hs[gidx])
                    pck_any_total += 1
                    best_d = float('inf')
                    # iterate all predictions for this image
                    for pidx in range(n_pred):
                        if pidx < kpts_pred.shape[0]:
                            d = float(np.linalg.norm((kpts_pred[pidx].cpu().numpy() - gt_kpts[gidx].numpy())))
                            if d < best_d:
                                best_d = d
                    if best_d <= thr:
                        pck_any_hits += 1
            # break outer loop if image limit reached
            if max_images is not None and images_processed >= max_images:
                break
        # compute AP (11‑point interpolation)
        if not all_scores or total_gt == 0:
            map50 = 0.0
        else:
            scores_arr = np.array(all_scores, dtype=np.float64)
            matches_arr = np.array(all_matches, dtype=np.int32)
            order_all = np.argsort(-scores_arr)
            matches_sorted = matches_arr[order_all]
            tp = matches_sorted
            fp = 1 - matches_sorted
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            denom = np.maximum(tp_cum + fp_cum, 1)
            prec = tp_cum / denom
            rec = tp_cum / max(1, total_gt)
            ap = 0.0
            for r in np.linspace(0, 1, 11):
                mask = rec >= r
                p = prec[mask].max() if np.any(mask) else 0.0
                ap += p / 11.0
            map50 = float(ap)
        # PCK
        pck = float(pck_hits / max(1, pck_total))
        # PCK_any
        pck_any = float(pck_any_hits / max(1, pck_any_total))
        # average predictions per image
        pred_per_img = float(total_preds / max(1, total_images))
        # recall at different thresholds
        recall_stats = {}
        for thr in self.recall_thresholds:
            recall_stats[thr] = float(recall_matches[thr] / max(1, recall_total))
        # for compatibility with common naming conventions, map is identical to map50 here
        metrics = {
            "mAP50": map50,
            "mAP": map50,
            f"pck@{self.pck_tau}": pck,
            f"pck_any@{self.pck_tau}": pck_any,
            "tps": int(np.sum(all_matches)),
            "pred_per_img": pred_per_img,
            "images": total_images,
            "best_iou": min(best_iou_global, 1.0),
        }
        # append recall statistics
        for thr in self.recall_thresholds:
            metrics[f"recall@{thr}"] = recall_stats[thr]
        return metrics