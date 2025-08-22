# src/engine/evaluator.py
from __future__ import annotations

import logging, math, os, traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    from mmcv.ops import box_iou_rotated as mmcv_box_iou_rotated

    _MMCV_OK = True
except Exception:
    _MMCV_OK = False

try:
    from shapely.geometry import Polygon

    _SHAPELY_OK = True
except Exception:
    _SHAPELY_OK = False

DEBUG_EVAL = os.environ.get("EVAL_DEBUG", "0") not in ("0", "", "false", "False", "no", "No")


def _cfg_to_params(cfg: Any, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    d = dict(conf_thres=0.25, iou_thres=0.70, multi_label=False, agnostic=False, use_nms=True, max_det=300,
             score_thresh=None, map_iou_st=0.50, map_iou_ed=0.95, map_iou_step=0.05, pck_alpha=0.05,
             topk_kpt=128, print_every=0)

    def to_dict(x):
        if x is None: return {}
        if isinstance(x, dict): return dict(x)
        out = {}
        for k in dir(x):
            if k.startswith("_"): continue
            try:
                v = getattr(x, k)
            except Exception:
                continue
            if callable(v): continue
            out[k] = v
        return out

    src = None
    if cfg is not None:
        for key in ("evaluator",):
            if isinstance(cfg, dict) and key in cfg: src = cfg[key]; break
            if hasattr(cfg, key): src = getattr(cfg, key); break
        if src is None: src = cfg
        for k, v in to_dict(src).items():
            if k in d: d[k] = v
    if params:
        for k, v in params.items():
            if k in d: d[k] = v
    for k in ("conf_thres", "iou_thres", "pck_alpha"): d[k] = float(d[k])
    for k in ("multi_label", "agnostic", "use_nms"): d[k] = bool(d[k])
    for k in ("max_det", "topk_kpt", "print_every"): d[k] = int(d[k])
    if d["score_thresh"] is not None: d["conf_thres"] = float(d["score_thresh"])
    del d["score_thresh"]
    d["map_iou_st"] = float(d["map_iou_st"])
    d["map_iou_ed"] = float(d["map_iou_ed"])
    d["map_iou_step"] = float(d["map_iou_step"])
    return d


def _get_first_batch(dl):
    it = iter(dl)
    batch = next(it)
    if isinstance(batch, dict):
        for key in ("image", "imgs", "images", "img"):
            if key in batch and batch[key] is not None:
                return batch[key]
        raise RuntimeError("Could not locate image tensor in batch dict; keys: %s" % list(batch.keys()))
    elif isinstance(batch, (list, tuple)):
        return batch[0]
    else:
        return batch


def _shape(x):
    try:
        if torch.is_tensor(x): return tuple(x.shape)
        if isinstance(x, (list, tuple)): return f"list(len={len(x)})"
        return type(x).__name__
    except Exception:
        return "?"


class Evaluator:
    def __init__(self, cfg: Optional[Any] = None, names: Optional[List[str]] = None,
                 logger: Optional[logging.Logger] = None, params: Optional[Dict[str, Any]] = None):
        self.log = logger or logging.getLogger("evaluator")
        self.names = names or []
        self.cfg = _cfg_to_params(cfg, params)
        self.iou_thrs = np.arange(self.cfg["map_iou_st"], self.cfg["map_iou_ed"] + 1e-9,
                                  self.cfg["map_iou_step"]).round(2)

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, val_loader, device: torch.device, max_images: Optional[int] = None) -> \
    Dict[str, float]:
        try:
            _ = _get_first_batch(val_loader)
        except StopIteration:
            raise RuntimeError("Validation DataLoader is empty.")

        model_eval = model.module if hasattr(model, "module") else model
        model_eval.eval()

        stats = {
            "tp_by_thr": {float(t): [] for t in self.iou_thrs},
            "fp_by_thr": {float(t): [] for t in self.iou_thrs},
            "scores_by_thr": {float(t): [] for t in self.iou_thrs},
            "gt_total": 0, "pred_count": 0, "best_iou": [],
            "recall_hits": {0.1: 0, 0.3: 0, 0.5: 0},
            "images_eval": 0, "pck_ok": 0, "pck_any_ok": 0, "pck_total": 0
        }

        img_seen = 0
        for batch in val_loader:
            if max_images is not None and img_seen >= max_images: break
            if not isinstance(batch, dict) or ("image" not in batch):
                self.log.warning("[eval] Unexpected batch format; skipping.")
                continue

            imgs = batch["image"].to(device, non_blocking=True).float()
            B = imgs.shape[0]
            outs = model(imgs)
            if isinstance(outs, dict):
                det_maps, feats = outs.get("det"), outs.get("feats")
            elif isinstance(outs, (list, tuple)):
                det_maps, feats = outs[0], (outs[1] if len(outs) > 1 else None)
            else:
                det_maps, feats = outs, None

            try:
                preds_list = model_eval.decode_obb_from_pyramids(
                    det_maps, imgs,
                    conf_thres=self.cfg["conf_thres"],
                    iou_thres=self.cfg["iou_thres"],
                    multi_label=self.cfg["multi_label"],
                    agnostic=self.cfg["agnostic"],
                    max_det=self.cfg["max_det"],
                    use_nms=self.cfg["use_nms"],
                )
            except TypeError:
                preds_list = model_eval.decode_obb_from_pyramids(
                    det_maps, imgs,
                    score_thr=self.cfg["conf_thres"],
                    max_det=self.cfg["max_det"],
                    use_nms=self.cfg["use_nms"],
                )
            except Exception as e:
                if DEBUG_EVAL:
                    self.log.error("[eval] decode failed with: %s\n%s", e, traceback.format_exc())
                else:
                    self.log.error("[eval] decode_obb_from_pyramids failed: %s", str(e))
                preds_list = None

            if preds_list is None or len(preds_list) != B:
                if DEBUG_EVAL:
                    self.log.error("[eval] decode returned invalid preds: %s (B=%d)", type(preds_list).__name__, B)
                preds_list = [{"boxes": torch.zeros((0, 5), device=device),
                               "scores": torch.zeros((0,), device=device),
                               "labels": torch.zeros((0,), dtype=torch.long, device=device)} for _ in range(B)]

            gtb_list = batch.get("bboxes", [torch.zeros((0, 5), device=device) for _ in range(B)])
            gtk_list = batch.get("kpts", [torch.zeros((0, 2), device=device) for _ in range(B)])
            gtl_list = batch.get("labels", [None for _ in range(B)])

            uv_img_by_b = [torch.empty((0, 2), device=device) for _ in range(B)]
            if hasattr(model_eval, "kpt_from_obbs") and feats is not None:
                try:
                    obb_for_kpt = [pred["boxes"].clone() if ("boxes" in pred) else pred.get("obb", torch.zeros((0, 5),
                                                                                                               device=device))
                                   for pred in preds_list]
                    scores_for_kpt = [pred.get("scores") for pred in preds_list]
                    uv_all, metas = model_eval.kpt_from_obbs(feats, obb_for_kpt, scores_list=scores_for_kpt,
                                                             topk=min(self.cfg["topk_kpt"], self.cfg["max_det"]),
                                                             chunk=128, score_thresh=self.cfg["conf_thres"])
                    if uv_all is not None and len(metas) == uv_all.shape[0]:
                        feat_down = getattr(getattr(model_eval, "roi", None), "feat_down", 8)
                        for (uv, meta) in zip(uv_all, metas):
                            if not isinstance(meta, dict) or ("M" not in meta) or ("bix" not in meta): continue
                            M = meta["M"]
                            bix = int(meta["bix"]) if isinstance(meta["bix"], int) else int(meta["bix"].item())
                            if not torch.is_tensor(M): M = torch.as_tensor(M, dtype=torch.float32, device=device)
                            M = M.to(device).float()
                            x = uv[0] * 2.0 - 1.0
                            y = uv[1] * 2.0 - 1.0
                            xy1 = torch.stack([x, y, torch.tensor(1.0, device=device)], dim=0)
                            xy_feat = (M @ xy1)
                            xy_img = xy_feat * float(feat_down)
                            uv_img_by_b[bix] = torch.cat([uv_img_by_b[bix], xy_img[None, :]], dim=0)
                except Exception as e:
                    if DEBUG_EVAL:
                        self.log.error("[eval] kpt_from_obbs failed with: %s\n%s", e, traceback.format_exc())
                    else:
                        self.log.warning("[eval] kpt_from_obbs failed: %s", str(e))

            for b in range(B):
                try:
                    img_seen += 1
                    stats["images_eval"] += 1
                    gt_boxes = gtb_list[b].to(device)
                    gt_kpts = gtk_list[b].to(device)
                    gt_labels = None
                    if gtl_list[b] is not None:
                        gt_labels = torch.as_tensor(gtl_list[b], device=device, dtype=torch.long).view(-1)

                    preds = preds_list[b]
                    boxes = preds["boxes"].to(device).float()
                    scores = preds["scores"].to(device).float().view(-1)
                    labels = preds.get("labels", None)
                    if labels is not None:
                        labels = torch.as_tensor(labels, device=device, dtype=torch.long).view(-1)

                    stats["pred_count"] += int(boxes.shape[0])
                    stats["gt_total"] += int(gt_boxes.shape[0])

                    iou_mat = self._iou_matrix(boxes, gt_boxes)
                    stats["best_iou"].append(float(iou_mat.max().item()) if iou_mat.numel() else 0.0)

                    order = torch.argsort(scores, descending=True)
                    used = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=boxes.device)

                    for thr in self.iou_thrs:
                        t = float(thr)
                        tp, fp = [], []
                        for i in order.tolist():
                            if gt_boxes.shape[0] == 0:
                                tp.append(0)
                                fp.append(1)
                                continue

                            has_cls = (labels is not None) and (gt_labels is not None) and (int(gt_labels.numel()) > 0)
                            if has_cls:
                                li = labels[i].view(-1)[0]
                                same_class = (gt_labels == li)
                                if not bool(same_class.any().item()):
                                    tp.append(0)
                                    fp.append(1)
                                    continue
                                cand_idx = same_class.nonzero(as_tuple=True)[0]
                                if iou_mat.numel() and cand_idx.numel():
                                    ious = iou_mat[i, cand_idx]
                                    j_rel = int(torch.argmax(ious).item()) if ious.numel() else 0
                                    j = int(cand_idx[j_rel].item()) if cand_idx.numel() else 0
                                    iou_ij = float(ious[j_rel].item()) if ious.numel() else 0.0
                                else:
                                    j, iou_ij = 0, 0.0
                            else:
                                if iou_mat.numel():
                                    j = int(torch.argmax(iou_mat[i]).item())
                                    iou_ij = float(iou_mat[i, j].item())
                                else:
                                    j, iou_ij = 0, 0.0

                            if (iou_ij >= t) and (not bool(used[j].item())):
                                tp.append(1)
                                fp.append(0)
                                used[j] = True
                            else:
                                fp.append(1)
                                tp.append(0)
                        stats["tp_by_thr"][t].extend(tp)
                        stats["fp_by_thr"][t].extend(fp)
                        stats["scores_by_thr"][t].extend([float(s) for s in scores[order].tolist()])
                        used.zero_()

                    if gt_boxes.shape[0] and boxes.shape[0] and iou_mat.numel():
                        max_per_gt = iou_mat.max(dim=0).values
                        for r in (0.1, 0.3, 0.5):
                            stats["recall_hits"][r] += int((max_per_gt >= r).sum().item())

                    uv_img = uv_img_by_b[b]
                    if uv_img.numel() and gt_kpts.numel():
                        sizes = torch.sqrt(boxes[:, 2] * boxes[:, 3]).clamp(min=1.0)
                        n = uv_img.shape[0]
                        thrv = (self.cfg["pck_alpha"] * (
                            sizes[:n] if sizes.numel() >= n else sizes[-1].repeat(n))) if sizes.numel() else torch.full(
                            (n,), 1.0, device=device)
                        dists = torch.cdist(uv_img.float(), gt_kpts.float())
                        if dists.numel():
                            dmin, _ = dists.min(dim=1)
                            ok = (dmin <= thrv)
                            stats["pck_ok"] += int(ok.sum().item())
                            stats["pck_any_ok"] += int((dmin <= thrv.max()).sum().item())
                            stats["pck_total"] += int(n)

                    if self.cfg["print_every"] and (img_seen % self.cfg["print_every"] == 0):
                        mn = float(scores.min().item()) if boxes.numel() else 0.0
                        me = float(scores.mean().item()) if boxes.numel() else 0.0
                        mx = float(scores.max().item()) if boxes.numel() else 0.0
                        self.log.info("[EVAL img] #%d GT=%d pred=%d bestIoU=%.3f score[min/mean/max]=[%.3f/%.3f/%.3f]",
                                      img_seen, int(gt_boxes.shape[0]), int(boxes.shape[0]),
                                      stats["best_iou"][-1], mn, me, mx)

                except Exception as e:
                    msg = (
                        f"[eval] per-image exception: {e}\n"
                        f"  boxes: {_shape(preds['boxes'])}  scores: {_shape(preds['scores'])}  labels: {_shape(preds.get('labels'))}\n"
                        f"  gt_boxes: {_shape(gt_boxes)}  gt_labels: {_shape(gt_labels)}  gt_kpts: {_shape(gt_kpts)}\n"
                        f"  iou_mat: {_shape(locals().get('iou_mat', None))}\n"
                        f"{traceback.format_exc()}"
                    )
                    if DEBUG_EVAL:
                        self.log.error(msg)
                    else:
                        self.log.error("[eval] per-image error (set EVAL_DEBUG=1 for details): %s", e)
                    # Skip this image and continue

        return self._finalize_and_log(stats)

    def _iou_matrix(self, boxes: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:

        use_aabb = bool(int(os.getenv("EVAL_USE_AABB", "0")))
        if use_aabb:
            return self._iou_aabb_pair(boxes, gts).clamp_(0, 1).to(boxes.device)

        device = boxes.device if torch.is_tensor(boxes) else torch.device("cpu")
        N = int(boxes.shape[0])
        M = int(gts.shape[0])
        if N == 0 or M == 0:
            return torch.zeros((N, M), device=device)

        if _MMCV_OK:
            b = boxes.detach().clone()
            g = gts.detach().clone()
            # mmcv uses degrees
            b[:, 4] = torch.rad2deg(b[:, 4])
            g[:, 4] = torch.rad2deg(g[:, 4])
            try:
                iou = mmcv_box_iou_rotated(b, g, aligned=False).detach().float().clamp_(0, 1)
                return iou.to(device)
            except Exception:
                pass

        if _SHAPELY_OK:
            return self._iou_shapely(boxes, gts).clamp_(0, 1).to(device)

        # Fallback AABB approximation (very last resort)
        return self._iou_aabb_pair(boxes, gts).clamp_(0, 1).to(device)

    def _to_poly(self, obb_np: np.ndarray):
        out = []
        for cx, cy, w, h, ang in obb_np:
            ca, sa = math.cos(float(ang)), math.sin(float(ang))
            dx, dy = w / 2.0, h / 2.0
            pts = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
            R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
            rot = pts @ R.T
            rot[:, 0] += cx
            rot[:, 1] += cy
            try:
                from shapely.geometry import Polygon
                out.append(Polygon(rot))
            except Exception:
                # if shapely is not available, we still return the points array
                out.append(rot)
        return out

    def _iou_shapely(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Compute rotated IoU via shapely polygons; falls back silently if shapely absent
        try:
            from shapely.geometry import Polygon
        except Exception:
            return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)
        an = a.detach().cpu().numpy()
        bn = b.detach().cpu().numpy()
        ap = self._to_poly(an)
        bp = self._to_poly(bn)
        out = np.zeros((an.shape[0], bn.shape[0]), dtype=np.float32)
        for i, pa in enumerate(ap):
            for j, pb in enumerate(bp):
                if not isinstance(pa, Polygon) or not isinstance(pb, Polygon):
                    continue
                inter = pa.intersection(pb).area
                if inter <= 0: continue
                u = pa.area + pb.area - inter
                out[i, j] = float(inter / u) if u > 0 else 0.0
        return torch.from_numpy(out)

    def _iou_aabb_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Axis-aligned fallback IoU; used only if mmcv/shapely unavailable
        def aabb(x):
            cx, cy, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
            return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

        a1, a2, a3, a4 = aabb(a)
        b1, b2, b3, b4 = aabb(b)
        A, B = a.shape[0], b.shape[0]
        out = torch.zeros((A, B), dtype=torch.float32)
        for i in range(A):
            for j in range(B):
                xx1 = max(float(a1[i]), float(b1[j]))
                yy1 = max(float(a2[i]), float(b2[j]))
                xx2 = min(float(a3[i]), float(b3[j]))
                yy2 = min(float(a4[i]), float(b4[j]))
                iw = max(0.0, xx2 - xx1)
                ih = max(0.0, yy2 - yy1)
                inter = iw * ih
                ua = (float(a3[i] - a1[i]) * float(a4[i] - a2[i]))
                ub = (float(b3[j] - b1[j]) * float(b4[j] - b2[j]))
                union = ua + ub - inter
                out[i, j] = float(inter / union) if union > 0 else 0.0
        return out

    def _ap_from_pr(self, tp: np.ndarray, fp: np.ndarray, sc: np.ndarray) -> float:
        if tp.size == 0: return 0.0
        order = np.argsort(-sc)
        tp = tp[order].astype(np.float32)
        fp = fp[order].astype(np.float32)
        ctp = np.cumsum(tp)
        cfp = np.cumsum(fp)
        denom = max(1.0, float(tp.sum()))
        recall = ctp / denom
        precision = ctp / np.maximum(ctp + cfp, 1e-9)
        rec_points = np.linspace(0, 1, 101)
        prec_at_rec = np.zeros_like(rec_points)
        for i, r in enumerate(rec_points):
            inds = np.where(recall >= r)[0]
            if inds.size: prec_at_rec[i] = precision[inds].max()
        return float(prec_at_rec.mean())

    def _finalize_and_log(self, st: dict) -> dict:
        aps = []
        for t in self.iou_thrs:
            t = float(t)
            tp = np.array(st["tp_by_thr"][t], dtype=np.float32)
            fp = np.array(st["fp_by_thr"][t], dtype=np.float32)
            sc = np.array(st["scores_by_thr"][t], dtype=np.float32)
            aps.append(self._ap_from_pr(tp, fp, sc))
        mAP = float(np.mean(aps)) if aps else 0.0
        try:
            i50 = list(self.iou_thrs).index(0.5)
            mAP50 = float(aps[i50]) if aps else 0.0
        except ValueError:
            mAP50 = 0.0

        images = float(st["images_eval"]) or 1.0
        pred_per_img = float(st["pred_count"]) / images
        gt_total = max(int(st["gt_total"]), 1)

        rec01 = float(st["recall_hits"][0.1]) / gt_total
        rec03 = float(st["recall_hits"][0.3]) / gt_total
        rec05 = float(st["recall_hits"][0.5]) / gt_total

        best_iou = float(np.mean(st["best_iou"])) if st["best_iou"] else 0.0

        if st["pck_total"] > 0:
            pck = float(st["pck_ok"]) / float(st["pck_total"])
            pck_any = float(st["pck_any_ok"]) / float(st["pck_total"])
        else:
            pck = 0.0
            pck_any = 0.0

        metrics = {
            "mAP50": mAP50, "mAP": mAP,
            "pck@0.05": pck, "pck_any@0.05": pck_any,
            "tps": float(st["recall_hits"][0.5]), "pred_per_img": pred_per_img,
            "images": int(images), "best_iou": np.float64(best_iou),
            "recall@0.1": rec01, "recall@0.3": rec03, "recall@0.5": rec05,
        }
        logging.getLogger("evaluator").info(
            "[EVAL] images %.1f  mAP50 %.6f  mAP %.6f  pck@0.05 %.6f  pck_any@0.05 %.6f  "
            "tps %.1f  pred_per_img %.2f  recall@0.1 %.2f  recall@0.3 %.2f  recall@0.5 %.2f  best_iou %.3f",
            metrics.get("images", 0.0), metrics.get("mAP50", 0.0), metrics.get("mAP", 0.0),
            metrics.get("pck@0.05", 0.0), metrics.get("pck_any@0.05", 0.0),
            metrics.get("tps", 0.0), metrics.get("pred_per_img", 0.0),
            metrics.get("recall@0.1", 0.0), metrics.get("recall@0.3", 0.0), metrics.get("recall@0.5", 0.0),
            metrics.get("best_iou", 0.0),
        )
        return metrics


def _iou_matrix(self, boxes: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
    device = boxes.device if torch.is_tensor(boxes) else torch.device("cpu")
    N = int(boxes.shape[0])
    M = int(gts.shape[0])
    if N == 0 or M == 0:
        return torch.zeros((N, M), device=device)

    if _MMCV_OK:
        b = boxes.detach().clone()
        g = gts.detach().clone()
        # mmcv uses degrees
        b[:, 4] = torch.rad2deg(b[:, 4])
        g[:, 4] = torch.rad2deg(g[:, 4])
        try:
            iou = mmcv_box_iou_rotated(b, g, aligned=False).detach().float().clamp_(0, 1)
            return iou.to(device)
        except Exception:
            pass

    if _SHAPELY_OK:
        return self._iou_shapely(boxes, gts).clamp_(0, 1).to(device)

    # Fallback AABB approximation (very last resort)
    return self._iou_aabb_pair(boxes, gts).clamp_(0, 1).to(device)


def _to_poly(self, obb_np: np.ndarray) -> List["Polygon"]:
    out = []
    for cx, cy, w, h, ang in obb_np:
        ca, sa = math.cos(float(ang)), math.sin(float(ang))
        dx, dy = w / 2.0, h / 2.0
        pts = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
        R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
        rot = pts @ R.T
        rot[:, 0] += cx
        rot[:, 1] += cy
        out.append(Polygon(rot))
    return out


def _iou_shapely(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    an = a.detach().cpu().numpy()
    bn = b.detach().cpu().numpy()
    ap = self._to_poly(an)
    bp = self._to_poly(bn)
    out = np.zeros((an.shape[0], bn.shape[0]), dtype=np.float32)
    for i, pa in enumerate(ap):
        for j, pb in enumerate(bp):
            inter = pa.intersection(pb).area
            if inter <= 0: continue
            u = pa.area + pb.area - inter
            out[i, j] = float(inter / u) if u > 0 else 0.0
    return torch.from_numpy(out)


def _iou_aabb_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # axis-aligned fallback IoU; used only if mmcv/shapely unavailable
    def aabb(x):
        cx, cy, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    a1, a2, a3, a4 = aabb(a)
    b1, b2, b3, b4 = aabb(b)
    A, B = a.shape[0], b.shape[0]
    out = torch.zeros((A, B))
    for i in range(A):
        for j in range(B):
            xx1 = max(float(a1[i]), float(b1[j]))
            yy1 = max(float(a2[i]), float(b2[j]))
            xx2 = min(float(a3[i]), float(b3[j]))
            yy2 = min(float(a4[i]), float(b4[j]))
            iw = max(0.0, xx2 - xx1)
            ih = max(0.0, yy2 - yy1)
            inter = iw * ih
            ua = (float(a3[i] - a1[i]) * float(a4[i] - a2[i]))
            ub = (float(b3[j] - b1[j]) * float(b4[j] - b2[j]))
            union = ua + ub - inter
            out[i, j] = float(inter / union) if union > 0 else 0.0
    return out


def _ap_from_pr(self, tp: np.ndarray, fp: np.ndarray, sc: np.ndarray) -> float:
    if tp.size == 0: return 0.0
    order = np.argsort(-sc)
    tp = tp[order].astype(np.float32)
    fp = fp[order].astype(np.float32)
    ctp = np.cumsum(tp)
    cfp = np.cumsum(fp)
    denom = max(1.0, float(tp.sum()))
    recall = ctp / denom
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    rec_points = np.linspace(0, 1, 101)
    prec_at_rec = np.zeros_like(rec_points)
    for i, r in enumerate(rec_points):
        inds = np.where(recall >= r)[0]
        if inds.size: prec_at_rec[i] = precision[inds].max()
    return float(prec_at_rec.mean())


def _finalize_and_log(self, st: Dict[str, Any]) -> Dict[str, Any]:
    aps = []
    for t in self.iou_thrs:
        t = float(t)
        tp = np.array(st["tp_by_thr"][t], dtype=np.float32)
        fp = np.array(st["fp_by_thr"][t], dtype=np.float32)
        sc = np.array(st["scores_by_thr"][t], dtype=np.float32)
        aps.append(self._ap_from_pr(tp, fp, sc))
    mAP = float(np.mean(aps)) if aps else 0.0
    try:
        i50 = list(self.iou_thrs).index(0.5)
        mAP50 = float(aps[i50]) if aps else 0.0
    except ValueError:
        mAP50 = 0.0

    images = float(st["images_eval"]) or 1.0
    pred_per_img = float(st["pred_count"]) / images
    gt_total = max(int(st["gt_total"]), 1)

    rec01 = float(st["recall_hits"][0.1]) / gt_total
    rec03 = float(st["recall_hits"][0.3]) / gt_total
    rec05 = float(st["recall_hits"][0.5]) / gt_total

    best_iou = float(np.mean(st["best_iou"])) if st["best_iou"] else 0.0

    if st["pck_total"] > 0:
        pck = float(st["pck_ok"]) / float(st["pck_total"])
        pck_any = float(st["pck_any_ok"]) / float(st["pck_total"])
    else:
        pck = 0.0
        pck_any = 0.0

    metrics = {
        "mAP50": mAP50, "mAP": mAP,
        "pck@0.05": pck, "pck_any@0.05": pck_any,
        "tps": float(st["recall_hits"][0.5]),
        "pred_per_img": pred_per_img,
        "images": int(images),
        "best_iou": np.float64(best_iou),
        "recall@0.1": rec01, "recall@0.3": rec03, "recall@0.5": rec05,
    }
    logging.getLogger("evaluator").info(
        "[EVAL] images %.1f  mAP50 %.6f  mAP %.6f  pck@0.05 %.6f  pck_any@0.05 %.6f  "
        "tps %.1f  pred_per_img %.2f  recall@0.1 %.2f  recall@0.3 %.2f  recall@0.5 %.2f  best_iou %.3f",
        metrics.get("images", 0.0), metrics.get("mAP50", 0.0), metrics.get("mAP", 0.0),
        metrics.get("pck@0.05", 0.0), metrics.get("pck_any@0.05", 0.0),
        metrics.get("tps", 0.0), metrics.get("pred_per_img", 0.0),
        metrics.get("recall@0.1", 0.0), metrics.get("recall@0.3", 0.0), metrics.get("recall@0.5", 0.0),
        metrics.get("best_iou", 0.0),
    )
    return metrics


EvaluatorFull = Evaluator