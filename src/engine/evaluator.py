# src/engine/evaluator.py
from types import SimpleNamespace
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# ================================================================
# Utilities
# ================================================================
def _ns(x):
    return x if isinstance(x, SimpleNamespace) else SimpleNamespace(**(x or {}))

def _get(ns, key, default):
    try:
        v = getattr(ns, key)
    except Exception:
        return default
    return default if v is None else v

def _to_bool(x, default=False):
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(x)

def _class_names_from_cfg(cfg, nc: int):
    try:
        names = list(getattr(cfg.data, "names"))
        if len(names) == nc:
            return names
    except Exception:
        pass
    try:
        names = list(getattr(cfg.model, "names"))
        if len(names) == nc:
            return names
    except Exception:
        pass
    return [f"c{i}" for i in range(nc)]

def _has_rotated_ops():
    has_nms = hasattr(torchvision.ops, "nms_rotated") or hasattr(torch.ops.torchvision, "nms_rotated")
    has_iou = hasattr(torchvision.ops, "box_iou_rotated") or hasattr(torch.ops.torchvision, "box_iou_rotated")
    return has_nms and has_iou

def _box_iou_rotated_tv(obb_a: torch.Tensor, obb_b: torch.Tensor) -> torch.Tensor:
    if hasattr(torchvision.ops, "box_iou_rotated"):
        return torchvision.ops.box_iou_rotated(obb_a, obb_b)
    if hasattr(torch.ops.torchvision, "box_iou_rotated"):
        return torch.ops.torchvision.box_iou_rotated(obb_a, obb_b)
    # fallback
    return box_iou_rotated_fallback(obb_a, obb_b)

def _nms_rotated_tv(obb: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    if hasattr(torchvision.ops, "nms_rotated"):
        return torchvision.ops.nms_rotated(obb, scores, float(iou_thr))
    if hasattr(torch.ops.torchvision, "nms_rotated"):
        return torch.ops.torchvision.nms_rotated(obb, scores, float(iou_thr))
    # fallback
    return nms_rotated_fallback(obb, scores, float(iou_thr))



# ================================================================
# Geometry helpers (pure PyTorch) — rotated IoU & NMS
# Works even if torchvision rotated ops are missing.
# OBB format: (cx, cy, w, h, angle_deg)
# ================================================================
_EPS = 1e-6

def _obb_to_poly_xy(obb: torch.Tensor) -> torch.Tensor:
    """
    obb: (5,) [cx,cy,w,h,angle_deg]
    returns (4,2) polygon in (x,y) order, clockwise
    """
    cx, cy, w, h, ang = obb
    w = torch.clamp(w, min=_EPS)
    h = torch.clamp(h, min=_EPS)
    rad = ang * (torch.pi / 180.0)
    cos_a = torch.cos(rad)
    sin_a = torch.sin(rad)
    # local corners: (-w/2,-h/2), (w/2,-h/2), (w/2,h/2), (-w/2,h/2)
    dx = torch.tensor([ -0.5*w,  0.5*w,  0.5*w, -0.5*w ], device=obb.device, dtype=obb.dtype)
    dy = torch.tensor([ -0.5*h, -0.5*h,  0.5*h,  0.5*h ], device=obb.device, dtype=obb.dtype)
    x = cx + dx * cos_a - dy * sin_a
    y = cy + dx * sin_a + dy * cos_a
    return torch.stack([x, y], dim=1)  # (4,2)

def _poly_area(p: torch.Tensor) -> torch.Tensor:
    """
    p: (N,2) polygon vertices (x,y) in order, N>=3
    return scalar area >= 0
    """
    if p.numel() == 0 or p.shape[0] < 3:
        return p.new_tensor(0.0)
    x = p[:, 0]; y = p[:, 1]
    s = x * torch.roll(y, -1) - torch.roll(x, -1) * y
    return 0.5 * torch.abs(torch.sum(s))

def _line_intersect(p1, p2, q1, q2):
    """
    Segment p1->p2 with q1->q2, returns (intersects, point)
    Uses line-line intersection, assumes not parallel in normal cases.
    """
    x1, y1 = p1; x2, y2 = p2; x3, y3 = q1; x4, y4 = q2
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if torch.abs(den) < 1e-12:
        return False, p1  # parallel or almost; treat as no-cut
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / den
    return True, torch.stack([px, py])

def _inside(pt, a, b):
    # Check if pt is on the left of edge a->b (for clockwise clipper, this keeps inside consistently)
    return (b[0]-a[0])*(pt[1]-a[1]) - (b[1]-a[1])*(pt[0]-a[0]) <= 0.0

def _suth_hodg_clip(subject: torch.Tensor, clipper: torch.Tensor) -> torch.Tensor:
    """
    Sutherland–Hodgman polygon clipping for convex polygons.
    subject: (Ns,2), clipper: (Nc,2)
    returns: (Nm,2) possibly empty
    """
    if subject.numel() == 0 or subject.shape[0] < 3:
        return subject.new_zeros((0, 2))
    output = subject
    for i in range(clipper.shape[0]):
        A = clipper[i]
        B = clipper[(i + 1) % clipper.shape[0]]
        if output.numel() == 0 or output.shape[0] == 0:
            break
        input_list = output
        output = []
        for j in range(input_list.shape[0]):
            S = input_list[j]
            E = input_list[(j + 1) % input_list.shape[0]]
            Ein = _inside(E, A, B)
            Sin = _inside(S, A, B)
            if Ein:
                if not Sin:
                    inter_ok, I = _line_intersect(S, E, A, B)
                    if inter_ok:
                        output.append(I)
                output.append(E)
            elif Sin:
                inter_ok, I = _line_intersect(S, E, A, B)
                if inter_ok:
                    output.append(I)
        if len(output) == 0:
            return input_list.new_zeros((0, 2))
        output = torch.stack(output, dim=0)
    return output

def _riou_pair(obb1: torch.Tensor, obb2: torch.Tensor,
               poly1: torch.Tensor = None, poly2: torch.Tensor = None,
               area1: torch.Tensor = None, area2: torch.Tensor = None) -> torch.Tensor:
    """
    IoU between two rotated rects (scalar). Optional poly/area to speed repeated calls.
    """
    if poly1 is None:
        poly1 = _obb_to_poly_xy(obb1)
    if poly2 is None:
        poly2 = _obb_to_poly_xy(obb2)
    if area1 is None:
        area1 = _poly_area(poly1)
    if area2 is None:
        area2 = _poly_area(poly2)
    inter_poly = _suth_hodg_clip(poly1, poly2)
    inter = _poly_area(inter_poly)
    union = torch.clamp(area1 + area2 - inter, min=_EPS)
    return inter / union

def box_iou_rotated_fallback(obbs1: torch.Tensor, obbs2: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch NxM IoU for rotated boxes. O(N*M), uses convex polygon clipping.
    """
    N = obbs1.shape[0]; M = obbs2.shape[0]
    if N == 0 or M == 0:
        return torch.zeros((N, M), device=obbs1.device, dtype=obbs1.dtype)
    polys1 = [ _obb_to_poly_xy(obbs1[i]) for i in range(N) ]
    areas1 = [ _poly_area(polys1[i]) for i in range(N) ]
    polys2 = [ _obb_to_poly_xy(obbs2[j]) for j in range(M) ]
    areas2 = [ _poly_area(polys2[j]) for j in range(M) ]
    out = torch.zeros((N, M), device=obbs1.device, dtype=obbs1.dtype)
    for i in range(N):
        for j in range(M):
            inter_poly = _suth_hodg_clip(polys1[i], polys2[j])
            inter = _poly_area(inter_poly)
            union = torch.clamp(areas1[i] + areas2[j] - inter, min=_EPS)
            out[i, j] = inter / union
    return out

def nms_rotated_fallback(obb: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """
    Pure PyTorch rotated-NMS using the fallback IoU. O(N^2).
    """
    if obb.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=obb.device)
    order = torch.argsort(scores, descending=True)
    keep: List[int] = []
    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        cur = obb[i].unsqueeze(0)  # (1,5)
        rest = obb[order[1:]]      # (M,5)
        ious = box_iou_rotated_fallback(cur, rest)[0]  # (M,)
        mask = ious <= float(iou_thr)
        order = order[1:][mask]
    return torch.tensor(keep, dtype=torch.long, device=obb.device)


# ================================================================
# Decoding (OBB head)
# det_maps per level: (B, 7+nc, H, W) with channels:
#   [tx,ty,tw,th,sin,cos,obj, cls[0..nc-1]]
# ================================================================
@torch.no_grad()
def _decode_det(det_maps, strides, score_thresh=0.0, nms_iou=0.5, max_det=5000,
                topk=10000, apply_nms=False, rotated_nms=True, multiclass_mode="argmax",
                num_classes=1, topk_per_level=None):

    device = det_maps[0].device
    B = det_maps[0].shape[0]
    outs = []
    has_cls = num_classes > 1 and (det_maps[0].shape[1] >= 7 + num_classes)

    for b in range(B):
        boxes_all, obb_all, scores_all, labels_all = [], [], [], []

        for dm, s in zip(det_maps, strides):
            logits = dm[b]  # (C,H,W)
            H, W = logits.shape[1], logits.shape[2]

            tx = logits[0].sigmoid(); ty = logits[1].sigmoid()
            tw = logits[2].exp();     th = logits[3].exp()
            si = logits[4].tanh();    co = logits[5].tanh()
            obj= logits[6].sigmoid()

            if has_cls:
                cls_prob = logits[7:7+num_classes].permute(1,2,0).contiguous().sigmoid()  # (H,W,nc)
            else:
                cls_prob = None

            ys, xs = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij"
            )
            cx = (tx + xs) * s; cy = (ty + ys) * s
            w  = tw.clamp_min(1e-6) * s; h = th.clamp_min(1e-6) * s
            x1 = cx - 0.5*w; y1 = cy - 0.5*h; x2 = cx + 0.5*w; y2 = cy + 0.5*h
            ang = torch.atan2(si, co) * (180.0 / torch.pi)

            if has_cls:
                # score map for per-level ranking
                maxp, arg = cls_prob.max(dim=2)  # (H,W)
                sc = obj * maxp
                lab = arg
            else:
                sc = obj
                lab = torch.zeros_like(sc, dtype=torch.long)

            # pre-threshold
            keep = sc > float(score_thresh)
            if keep.any():
                # OPTIONAL: per-level topk
                if topk_per_level is not None:
                    k = min(int(topk_per_level), int(keep.sum().item()))
                    if k > 0:
                        flat_sc = sc[keep]
                        idx_k = torch.topk(flat_sc, k=k, largest=True, sorted=False).indices
                        # Build a mask limited to the top-k positions
                        mask = torch.zeros_like(flat_sc, dtype=torch.bool)
                        mask[idx_k] = True
                        # Remap to full keep mask
                        # get indices of keep
                        keep_idx = keep.nonzero(as_tuple=False)  # (M,2) [y,x]
                        # select those in idx_k
                        sel_idx = keep_idx[mask]
                        # build a final mask same shape as sc
                        final_keep = torch.zeros_like(sc, dtype=torch.bool)
                        final_keep[sel_idx[:,0], sel_idx[:,1]] = True
                        keep = final_keep

                boxes_all.append(torch.stack([x1[keep], y1[keep], x2[keep], y2[keep]], dim=-1))
                obb_all.append(torch.stack([cx[keep], cy[keep], w[keep], h[keep], ang[keep]], dim=-1))
                scores_all.append(sc[keep])
                labels_all.append(lab[keep])

        if not scores_all:
            outs.append({"boxes": torch.zeros((0,4), device=device),
                         "obb": torch.zeros((0,5), device=device),
                         "scores": torch.zeros((0,), device=device),
                         "labels": torch.zeros((0,), dtype=torch.long, device=device)})
            continue

        boxes = torch.cat(boxes_all, 0)
        obb   = torch.cat(obb_all, 0)
        scores= torch.cat(scores_all, 0)
        labels= torch.cat(labels_all, 0)

        # global pre-cap
        if topk is not None and scores.numel() > int(topk):
            idx = torch.topk(scores, k=int(topk), largest=True, sorted=True).indices
            boxes, obb, scores, labels = boxes[idx], obb[idx], scores[idx], labels[idx]

        if apply_nms:
            if rotated_nms:
                keep = _nms_rotated_tv(obb, scores, float(nms_iou))
            else:
                keep = torchvision.ops.nms(boxes, scores, float(nms_iou))
            if keep.numel() > max_det:
                keep = keep[:max_det]
            boxes, obb, scores, labels = boxes[keep], obb[keep], scores[keep], labels[keep]
        else:
            if scores.numel() > max_det:
                idx = torch.topk(scores, k=max_det, largest=True, sorted=True).indices
                boxes, obb, scores, labels = boxes[idx], obb[idx], scores[idx], labels[idx]

        outs.append({"boxes": boxes, "obb": obb, "scores": scores, "labels": labels})
    return outs

# ================================================================
# Evaluator
# ================================================================
def quads_to_obb(quads: torch.Tensor) -> torch.Tensor:
    """
    Convert (N,4,2) corners to rotated boxes (cx,cy,w,h,deg).
    """
    cx = quads[:, :, 0].mean(dim=1)
    cy = quads[:, :, 1].mean(dim=1)
    v01 = quads[:, 1] - quads[:, 0]
    v12 = quads[:, 2] - quads[:, 1]
    l01 = torch.linalg.vector_norm(v01, dim=1)
    l12 = torch.linalg.vector_norm(v12, dim=1)
    is_w = (l01 >= l12)
    w = torch.where(is_w, l01, l12)
    h = torch.where(is_w, l12, l01)
    v = torch.where(is_w.unsqueeze(1), v01, v12)
    ang = torch.atan2(v[:, 1], v[:, 0]) * (180.0 / torch.pi)
    return torch.stack([cx, cy, w, h, ang], dim=-1)

class Evaluator:
    """
    Computes:
      - COCO-style mAP@[.5:.95] (101-pt) + mAP@.50
      - PCK@0.05 (matched) and PCK_any@0.05 (no IoU gating)
    Works without torchvision rotated ops by using fallbacks above.
    Expects model to implement kpt_from_obbs(feats, obb_list) for top-down kpts.
    """

    def __init__(self, cfg, debug: bool = False):
        self.cfg = cfg
        ev = _ns(getattr(cfg, "eval", None))
        md = _ns(getattr(cfg, "model", None))
        dc = _ns(getattr(cfg, "decode", None))
        td = _ns(getattr(cfg, "topdown", None))

        iou_thrs = _get(ev, "iou_thrs", None)
        self.iou_thrs = np.arange(0.50, 0.96, 0.05) if iou_thrs is None else np.asarray(iou_thrs, dtype=np.float64)

        self.iou_type = str(_get(ev, "iou_type", "obb")).lower()  # "obb" or "aabb"
        self.score_thresh = float(_get(ev, "score_thresh", 0.0))
        self.base_iou_thr = float(_get(ev, "iou_thr", 0.50))
        self.pck_tau = float(_get(ev, "pck_tau", 0.05))
        self.max_det = int(_get(ev, "max_det", 5000))
        self.topk = int(_get(ev, "topk", 10000))
        self.rank_by = str(_get(ev, "rank_by", "score"))
        self.strides = tuple(_get(md, "strides", (8, 16, 32)))
        self.nc = int(_get(md, "num_classes", 1))
        self.print_table = _to_bool(_get(ev, "print_table", True))
        self.debug = bool(debug)
        self.names = _class_names_from_cfg(cfg, self.nc)

        self.dec_args = dict(
            strides=self.strides,
            score_thresh=float(_get(dc, "score_thresh", 0.0)),
            nms_iou=float(_get(dc, "nms_iou", 0.50)),
            max_det=int(_get(dc, "max_det", 5000)),
            topk=int(_get(dc, "topk", 10000)),
            topk_per_level=int(_get(dc, "topk_per_level", 400)),
            apply_nms=_to_bool(_get(dc, "apply_nms", False)),
            rotated_nms=_to_bool(_get(dc, "rotated_nms", True)),
            multiclass_mode=str(_get(dc, "multiclass_mode", "argmax")),
            num_classes=self.nc,
        )
        if not _has_rotated_ops():
            self.dec_args["rotated_nms"] = False  # use fast AABB NMS
            self.dec_args["topk"] = min(self.dec_args["topk"], 600)

        self.crop_size = int(_get(td, "crop_size", 64))
        self.feat_down = int(self.strides[0])  # crop from P3

    @torch.no_grad()
    def evaluate(self, model, loader, device: str = "cpu", max_images: int = None) -> Dict[str, float]:
        model.eval()

        images_seen = 0
        use_amp = torch.cuda.is_available()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=use_amp):
            for batch in loader:
                if (max_images is not None) and (images_seen >= max_images):
                    break
                images_seen += len(batch["image"])

        T = len(self.iou_thrs)
        cls_scores = [[[] for _ in range(self.nc)] for __ in range(T)]
        cls_matches = [[[] for _ in range(self.nc)] for __ in range(T)]
        cls_total_gt = [0 for _ in range(self.nc)]
        pck_hits = 0; pck_total = 0
        pck_any_hits = 0; pck_any_total = 0
        images_count = 0; tp_count_base = 0; pred_total = 0
        recall_hits_01 = 0; recall_hits_03 = 0; recall_hits_05 = 0; recall_total = 0
        best_iou_list = []

        for batch in loader:
            images_count += len(batch["image"])
            imgs = batch["image"].to(device)
            outs = model(imgs)

            preds = _decode_det(outs["det"], **self.dec_args)
            obb_list = [p["obb"] for p in preds]
            scores_list = [p["scores"] for p in preds]

            uv_all, metas = model.kpt_from_obbs(
                outs["feats"], obb_list,
                scores_list=scores_list,
                topk=int(getattr(self.cfg.topdown, "kpt_topk", 128)),
                chunk=int(getattr(self.cfg.topdown, "roi_chunk", 128)),
                score_thresh=float(getattr(self.cfg.topdown, "score_thresh_kpt", 0.0)),
            )

            pred_kpts_img = [[] for _ in range(len(imgs))]
            for j, meta in enumerate(metas):
                bix = int(meta["bix"])
                uv = uv_all[j]; M = meta["M"]
                # uv -> image xy
                uS = uv[0] * self.crop_size
                vS = uv[1] * self.crop_size
                x_feat = M[0, 0]*uS + M[0, 1]*vS + M[0, 2]
                y_feat = M[1, 0]*uS + M[1, 1]*vS + M[1, 2]
                x_img = x_feat * self.feat_down
                y_img = y_feat * self.feat_down
                pred_kpts_img[bix].append(torch.stack([x_img, y_img], dim=0))
            pred_kpts_img = [
                (torch.stack(lst, dim=0) if len(lst) else torch.zeros((0, 2), device=imgs.device))
                for lst in pred_kpts_img
            ]

            for i in range(len(imgs)):
                gt_labels = batch["labels"][i].to(device) if len(batch["labels"][i]) else torch.zeros((0,), dtype=torch.long, device=device)
                gt_boxes  = batch["boxes"][i].to(device) if len(batch["boxes"][i]) else torch.zeros((0, 4), device=device)
                gt_quads  = batch["quads"][i].to(device) if "quads" in batch and len(batch["quads"][i]) else None
                gt_angles = batch["angles"][i].to(device) if "angles" in batch and len(batch["angles"][i]) else None
                gt_kpts   = batch["kpts"][i].to(device)  if len(batch["kpts"][i]) else torch.zeros((0, 2), device=device)

                for c in gt_labels.tolist():
                    if 0 <= c < self.nc:
                        cls_total_gt[c] += 1

                pr = preds[i]
                pb, po, ps, pl = pr["boxes"], pr["obb"], pr["scores"], pr["labels"]
                pred_total += int(pb.shape[0])
                pk = pred_kpts_img[i]

                # Build GT OBBs
                if self.iou_type == "obb":
                    if gt_quads is not None and gt_quads.numel():
                        gt_obb = quads_to_obb(gt_quads)
                    elif gt_angles is not None and gt_angles.numel():
                        cx = 0.5*(gt_boxes[:,0] + gt_boxes[:,2])
                        cy = 0.5*(gt_boxes[:,1] + gt_boxes[:,3])
                        w  = (gt_boxes[:,2] - gt_boxes[:,0]).clamp_min(1.0)
                        h  = (gt_boxes[:,3] - gt_boxes[:,1]).clamp_min(1.0)
                        gt_obb = torch.stack([cx,cy,w,h, gt_angles*(180.0/torch.pi)], dim=-1)
                    else:
                        cx = 0.5*(gt_boxes[:,0] + gt_boxes[:,2])
                        cy = 0.5*(gt_boxes[:,1] + gt_boxes[:,3])
                        w  = (gt_boxes[:,2] - gt_boxes[:,0]).clamp_min(1.0)
                        h  = (gt_boxes[:,3] - gt_boxes[:,1]).clamp_min(1.0)
                        gt_obb = torch.stack([cx,cy,w,h, torch.zeros_like(cx)], dim=-1)
                else:
                    gt_obb = None

                # Global recall diagnostics
                if self.iou_type == "obb":
                    if po.numel() and gt_obb is not None and gt_obb.numel():
                        ious_full = _box_iou_rotated_tv(po, gt_obb)
                    else:
                        ious_full = torch.zeros((po.shape[0], gt_boxes.shape[0]), device=device)
                else:
                    ious_full = torchvision.ops.box_iou(pb, gt_boxes) if (pb.numel() and gt_boxes.numel()) else ...

                if ious_full.numel() and ious_full.shape[1] > 0:
                    best_iou_per_gt = ious_full.max(dim=0).values
                    best_iou_list.extend(best_iou_per_gt.detach().cpu().tolist())
                    recall_hits_01 += int((best_iou_per_gt >= 0.1).sum().item())
                    recall_hits_03 += int((best_iou_per_gt >= 0.3).sum().item())
                    recall_hits_05 += int((best_iou_per_gt >= 0.5).sum().item())
                    recall_total    += int(best_iou_per_gt.numel())

                # PCK_any (no IoU gate)
                if gt_kpts.numel() and pk.numel():
                    d = torch.cdist(pk, gt_kpts)
                    best = d.min(dim=0).values
                    if gt_boxes.numel():
                        w = (gt_boxes[:,2] - gt_boxes[:,0]).clamp_min(1.0)
                        h = (gt_boxes[:,3] - gt_boxes[:,1]).clamp_min(1.0)
                    elif self.iou_type == "obb" and gt_obb is not None and gt_obb.numel():
                        w = gt_obb[:,2]; h = gt_obb[:,3]
                    else:
                        w = torch.ones((gt_kpts.shape[0],), device=device)
                        h = torch.ones((gt_kpts.shape[0],), device=device)
                    thr = self.pck_tau * torch.maximum(w, h)
                    pck_any_hits += int((best <= thr).sum().item())
                    pck_any_total += int(gt_kpts.shape[0])

                # Class-wise matching for each IoU threshold
                for c in range(self.nc):
                    idx_p = (pl == c).nonzero(as_tuple=False).flatten()
                    idx_g = (gt_labels == c).nonzero(as_tuple=False).flatten()
                    if idx_p.numel() == 0:
                        continue
                    if self.rank_by == "iou" and idx_g.numel() > 0:
                        ranker = ious_full[idx_p][:, idx_g].max(dim=1).values
                    else:
                        ranker = ps[idx_p]
                    order = torch.argsort(ranker, descending=True)
                    Pidx = idx_p[order]
                    rank_sorted = ranker[order]

                    for t_idx, thr in enumerate(self.iou_thrs):
                        taken = torch.zeros((idx_g.numel(),), dtype=torch.bool, device=device)
                        for j, pidx in enumerate(Pidx):
                            cls_scores[t_idx][c].append(float(rank_sorted[j].item()))
                            if idx_g.numel() == 0:
                                cls_matches[t_idx][c].append(0)
                                continue
                            i_best = torch.argmax(ious_full[pidx, idx_g])
                            iou = ious_full[pidx, idx_g[i_best]]
                            if iou >= thr and not taken[i_best]:
                                cls_matches[t_idx][c].append(1)
                                taken[i_best] = True
                                if abs(thr - self.base_iou_thr) < 1e-6:
                                    tp_count_base += 1
                                    if gt_kpts.numel() and pk.numel():
                                        gind = idx_g[i_best]
                                        if gt_boxes.numel():
                                            w = (gt_boxes[gind,2] - gt_boxes[gind,0]).clamp_min(1.0)
                                            h = (gt_boxes[gind,3] - gt_boxes[gind,1]).clamp_min(1.0)
                                        else:
                                            w = gt_obb[gind,2]; h = gt_obb[gind,3]
                                        thr_px = self.pck_tau * torch.maximum(w, h)
                                        if pidx < pk.shape[0]:
                                            dist = torch.linalg.vector_norm(pk[pidx] - gt_kpts[gind])
                                            pck_hits += int(dist <= thr_px)
                                            pck_total += 1
                            else:
                                cls_matches[t_idx][c].append(0)

        # AP/mAP
        T = len(self.iou_thrs)
        ap_thr_cls = np.zeros((T, self.nc), dtype=np.float64)
        ap50_per_class = np.zeros((self.nc,), dtype=np.float64)

        for t_idx, thr in enumerate(self.iou_thrs):
            for c in range(self.nc):
                scores = np.asarray(cls_scores[t_idx][c], dtype=np.float64)
                matches = np.asarray(cls_matches[t_idx][c], dtype=np.int32)
                tot_gt = int(cls_total_gt[c])
                if scores.size == 0 or tot_gt == 0:
                    ap_thr_cls[t_idx, c] = 0.0
                    continue
                order = np.argsort(-scores)
                m = matches[order]
                tp = m
                fp = 1 - m
                tp_cum = np.cumsum(tp)
                fp_cum = np.cumsum(fp)
                prec = tp_cum / np.maximum(tp_cum + fp_cum, 1)
                rec  = tp_cum / max(1, tot_gt)

                # 101-pt interpolation
                mrec = np.concatenate(([0.0], rec, [1.0]))
                mpre = np.concatenate(([0.0], prec, [0.0]))
                for i in range(mpre.size - 1, 0, -1):
                    mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
                recall_levels = np.linspace(0.0, 1.0, 101)
                ap = 0.0
                for r in recall_levels:
                    p = mpre[mrec >= r].max() if np.any(mrec >= r) else 0.0
                    ap += p
                ap /= 101.0
                ap_thr_cls[t_idx, c] = ap
            if abs(thr - 0.50) < 1e-6:
                ap50_per_class = ap_thr_cls[t_idx].copy()

        map_per_thr = ap_thr_cls.mean(axis=1) if self.nc > 0 else np.zeros((T,), dtype=np.float64)
        map_all = float(map_per_thr.mean()) if T > 0 else 0.0
        map50 = float(ap50_per_class.mean()) if self.nc > 0 else 0.0
        ap_per_class = ap_thr_cls.mean(axis=0).tolist()
        ap50_per_class = ap50_per_class.tolist()

        metrics = {
            "map": map_all,
            "map50": map50,
            "ap_per_class": ap_per_class,
            "ap50_per_class": ap50_per_class,
            "iou_thrs": self.iou_thrs.tolist(),
            "pck@0.05": float(pck_hits / max(1, pck_total)),
            "pck_any@0.05": float(pck_any_hits / max(1, pck_any_total)),
            "images": images_count,
            "tp_count@base": int(tp_count_base),
            "pred_per_img_avg": float(pred_total / max(1, images_count)),
            "recall@0.1": float(recall_hits_01 / max(1, recall_total)),
            "recall@0.3": float(recall_hits_03 / max(1, recall_total)),
            "recall@0.5": float(recall_hits_05 / max(1, recall_total)),
            "best_iou_mean": float(np.mean(best_iou_list)) if best_iou_list else 0.0,
        }

        if self.print_table:
            self._print_table(metrics)

        if self.debug:
            print("[eval] metrics:", metrics)

        return metrics

    def _print_table(self, metrics: Dict[str, float]):
        names = self.names
        ap = metrics["ap_per_class"]
        ap50 = metrics["ap50_per_class"]
        nc = len(names)
        hdr = f"{'Class':<16} {'AP@50:95':>9} {'AP@50':>9}"
        print(hdr)
        for i in range(nc):
            print(f"{names[i]:<16} {ap[i]:>9.3f} {ap50[i]:>9.3f}")
        print(f"{'-'*36}")
        print(f"{'all':<16} {metrics['map']:>9.3f} {metrics['map50']:>9.3f}")
