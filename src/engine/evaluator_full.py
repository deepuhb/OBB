# src/engine/evaluator_full.py (simplified for YOLO11_OBBPOSE_TD)
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---- optional geometry backends ----
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


def _cfg_to_params(cfg: Any, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge (defaults <- cfg.evaluator or cfg <- params). Only known keys are used."""
    def to_dict(x):
        if x is None:
            return {}
        if isinstance(x, dict):
            return dict(x)
        out = {}
        for k in dir(x):
            if k.startswith("_"):
                continue
            try:
                v = getattr(x, k)
            except Exception:
                continue
            if callable(v):
                continue
            out[k] = v
        return out

    d = dict(
        score_thresh=0.25,
        max_det=100,
        map_iou_st=0.5, map_iou_ed=0.95, map_iou_step=0.05,
        pck_alpha=0.05,
        print_every=0,
    )

    src = None
    if cfg is not None:
        for key in ("evaluator",):
            if isinstance(cfg, dict) and key in cfg:
                src = cfg[key]; break
            if hasattr(cfg, key):
                src = getattr(cfg, key); break
        if src is None:
            src = cfg
        for k, v in to_dict(src).items():
            if k in d:
                d[k] = v
    if params:
        for k, v in params.items():
            if k in d:
                d[k] = v

    d["score_thresh"] = float(d["score_thresh"])
    d["max_det"] = int(d["max_det"])
    d["map_iou_st"] = float(d["map_iou_st"])
    d["map_iou_ed"] = float(d["map_iou_ed"])
    d["map_iou_step"] = float(d["map_iou_step"])
    d["pck_alpha"] = float(d["pck_alpha"])
    d["print_every"] = int(d["print_every"])
    return d


def _get_first_batch(dl):
    it = iter(dl)
    batch = next(it)
    if isinstance(batch, dict):
        imgs = batch.get("image") or batch.get("imgs") or batch.get("images") or batch.get("image")
        targets = batch.get("targets") or batch.get("labels")
    elif isinstance(batch, (list, tuple)):
        imgs = batch[0]
        targets = batch[1] if len(batch) > 1 else None
    else:
        imgs, targets = batch, None
    if imgs is None:
        raise RuntimeError("Could not locate images in the first batch; check your dataset return format.")
    return imgs, targets


class EvaluatorFull:
    """
    OBB + (optional) single-keypoint evaluator.

    Expects val batch like:
      {"image": (B,3,H,W),
       "bboxes": list[Ti,5]  (cx,cy,w,h,rad) in px,
       "labels": list[Ti],   ints,
       "kpts":   list[Ti,2]  (x,y) px, ...}

    Assumes model implements:
      - forward(images) -> {'det': [P3,P4,P5], 'feats': [P3,P4,P5]}
      - decode_obb_from_pyramids(det_maps, imgs, score_thr, max_det) -> list[{'boxes','scores','cls',...}]
      - (optional) kpt_from_obbs(feats, obb_list, scores_list=None, topk, chunk, score_thresh)
          returns (uv_all, metas) where uv_all in [0,1], metas[i]['M'] is 2x3 crop->image affine and 'bix' image index.
    """

    def __init__(self,
                 cfg: Optional[Any] = None,
                 names: Optional[List[str]] = None,
                 logger: Optional[logging.Logger] = None,
                 params: Optional[Dict[str, Any]] = None):
        self.log = logger or logging.getLogger("evaluator_full")
        self.names = names or []
        self.cfg = _cfg_to_params(cfg, params)

        self.iou_thrs = np.arange(
            self.cfg["map_iou_st"], self.cfg["map_iou_ed"] + 1e-9, self.cfg["map_iou_step"]
        ).round(2)

    # ---------------- EVALUATE ----------------

    @torch.no_grad()
    def evaluate(self,
                 model: torch.nn.Module,
                 val_loader: torch.utils.data.DataLoader,
                 device: torch.device,
                 max_images: Optional[int] = None) -> Dict[str, Any]:

        # sanity: loader not empty
        try:
            _ = _get_first_batch(val_loader)
        except StopIteration:
            raise RuntimeError("Validation DataLoader is empty. Check val dataset length / filters.")

        model_eval = model.module if hasattr(model, "module") else model
        model_eval.eval()

        stats = {
            "tp_by_thr": {float(t): [] for t in self.iou_thrs},
            "fp_by_thr": {float(t): [] for t in self.iou_thrs},
            "scores_by_thr": {float(t): [] for t in self.iou_thrs},
            "gt_total": 0, "pred_count": 0, "best_iou": [],
            "recall_hits": {0.1: 0, 0.3: 0, 0.5: 0},
            "images_eval": 0,
            "pck_ok": 0, "pck_any_ok": 0, "pck_total": 0
        }

        img_seen = 0
        for batch in val_loader:
            if max_images is not None and img_seen >= max_images:
                break
            if not isinstance(batch, dict) or "image" not in batch:
                self.log.warning("[eval] Unexpected batch format; skipping.")
                continue

            imgs = batch["image"].to(device, non_blocking=True).float()
            B, _, H, W = imgs.shape

            outs = model(imgs)
            det_maps = outs["det"] if isinstance(outs, dict) else (outs[0] if isinstance(outs, (list,tuple)) else outs)
            feats = outs.get("feats") if isinstance(outs, dict) else (outs[1] if isinstance(outs, (list,tuple)) and len(outs) > 1 else None)

            # ---- decode via the model's canonical decoder ----
            try:
                preds_list = model_eval.decode_obb_from_pyramids(det_maps, imgs,
                                                                 score_thr=self.cfg["score_thresh"],
                                                                 max_det=self.cfg["max_det"])
            except Exception as e:
                self.log.error("[eval] decode_obb_from_pyramids failed: %s", str(e))
                preds_list = None

            if preds_list is None or len(preds_list) != B:
                self.log.error("[eval] Could not decode predictions; returning empty preds.")
                preds_list = [{"boxes": torch.zeros((0,5), device=device),
                               "scores": torch.zeros((0,), device=device),
                               "cls": None} for _ in range(B)]

            # ---- ground truth ----
            gtb_list = batch.get("bboxes", [torch.zeros((0, 5), device=device) for _ in range(B)])
            gtk_list = batch.get("kpts",   [torch.zeros((0, 2), device=device) for _ in range(B)])

            # ---- top-down keypoints (optional) ----
            uv_img_by_b = [torch.empty((0,2), device=device) for _ in range(B)]
            if hasattr(model_eval, "kpt_from_obbs") and feats is not None:
                try:
                    obb_for_kpt = [pred.get("obb") if (isinstance(pred, dict) and "obb" in pred) else pred["boxes"].clone()
                                   for pred in preds_list]
                    scores_for_kpt = [pred.get("scores") for pred in preds_list]
                    uv_all, metas = model_eval.kpt_from_obbs(feats, obb_for_kpt, scores_list=scores_for_kpt,
                                                             topk=min(128, self.cfg["max_det"]),
                                                             chunk=128, score_thresh=self.cfg["score_thresh"])
                    if uv_all is not None and len(metas) == uv_all.shape[0]:
                        # map each (u,v) in [0,1] to image pixels via affine M (2x3)
                        for (uv, meta) in zip(uv_all, metas):
                            if not isinstance(meta, dict) or ("M" not in meta) or ("bix" not in meta):
                                continue
                            M = meta["M"]
                            bix = int(meta["bix"]) if isinstance(meta["bix"], (int,)) else int(meta["bix"].item())
                            if not torch.is_tensor(M):
                                M = torch.as_tensor(M, dtype=torch.float32, device=device)
                            M = M.to(device).float()  # (2,3)
                            # normalize uv -> [-1,1] assumed by affine
                            x = uv[0] * 2.0 - 1.0
                            y = uv[1] * 2.0 - 1.0
                            xy1 = torch.stack([x, y, torch.tensor(1.0, device=device)], dim=0)  # (3,)
                            xy_img = (M @ xy1)  # (2,)
                            uv_img_by_b[bix] = torch.cat([uv_img_by_b[bix], xy_img[None, :]], dim=0)
                except Exception as e:
                    self.log.warning("[eval] kpt_from_obbs failed: %s", str(e))

            # ---- per-image scoring ----
            for b in range(B):
                img_seen += 1
                stats["images_eval"] += 1
                gt_boxes = gtb_list[b].to(device)
                gt_kpts  = gtk_list[b].to(device)
                preds    = preds_list[b]

                boxes  = preds["boxes"]
                scores = preds["scores"]

                stats["pred_count"] += int(boxes.shape[0])
                stats["gt_total"]   += int(gt_boxes.shape[0])

                iou_mat = self._iou_matrix(boxes, gt_boxes)
                stats["best_iou"].append(float(iou_mat.max().item()) if iou_mat.numel() else 0.0)

                order = torch.argsort(scores, descending=True)
                used = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=boxes.device)
                for t in self.iou_thrs:
                    tp, fp = [], []
                    for i in order.tolist():
                        if gt_boxes.shape[0] == 0:
                            tp.append(0); fp.append(1); continue
                        j = int(torch.argmax(iou_mat[i])) if iou_mat.numel() else 0
                        iou_ij = float(iou_mat[i, j].item()) if iou_mat.numel() else 0.0
                        if iou_ij >= float(t) and not bool(used[j]):
                            tp.append(1); fp.append(0); used[j] = True
                        else:
                            tp.append(0); fp.append(1)
                    stats["tp_by_thr"][float(t)].extend(tp)
                    stats["fp_by_thr"][float(t)].extend(fp)
                    stats["scores_by_thr"][float(t)].extend([float(s) for s in scores[order].tolist()])
                    used.zero_()

                if gt_boxes.shape[0] and boxes.shape[0] and iou_mat.numel():
                    max_per_gt = iou_mat.max(dim=0).values
                    for r in (0.1, 0.3, 0.5):
                        stats["recall_hits"][r] += int((max_per_gt >= r).sum().item())

                # PCK on predicted keypoints for this image (if any)
                uv_img = uv_img_by_b[b]
                if uv_img.numel() and gt_kpts.numel():
                    sizes = torch.sqrt(boxes[:, 2] * boxes[:, 3]).clamp(min=1.0)
                    # if fewer kpts than boxes, use matched sizes; else broadcast first K sizes
                    n = uv_img.shape[0]
                    if sizes.numel() == 0:
                        thr = torch.full((n,), 1.0, device=device)
                    else:
                        thr = self.cfg["pck_alpha"] * (sizes[:n] if sizes.numel() >= n else sizes[-1].repeat(n))
                    dists = torch.cdist(uv_img.float(), gt_kpts.float())  # (n, G)
                    if dists.numel():
                        dmin, _ = dists.min(dim=1)
                        ok = (dmin <= thr)
                        stats["pck_ok"] += int(ok.sum().item())
                        stats["pck_any_ok"] += int((dmin <= thr.max()).sum().item())
                        stats["pck_total"] += int(n)

                if self.cfg["print_every"] and (img_seen % self.cfg["print_every"] == 0):
                    self.log.info("[EVAL img] #%d GT=%d pred=%d bestIoU=%.3f score[min/mean/max]=[%.3f/%.3f/%.3f]",
                                  img_seen, int(gt_boxes.shape[0]), int(boxes.shape[0]),
                                  stats["best_iou"][-1],
                                  float(scores.min().item()) if boxes.numel() else 0.0,
                                  float(scores.mean().item()) if boxes.numel() else 0.0,
                                  float(scores.max().item()) if boxes.numel() else 0.0)

        metrics = self._finalize(stats)
        self._pretty(metrics)
        return metrics

    # --------------- IoU ---------------
    def _iou_matrix(self, boxes: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
        N = int(boxes.shape[0]); M = int(gts.shape[0])
        if N == 0 or M == 0:
            return torch.zeros((N, M))

        if _MMCV_OK:
            # mmcv expects degrees
            b = boxes.clone(); g = gts.clone()
            b[:, 4] = torch.rad2deg(b[:, 4])
            g[:, 4] = torch.rad2deg(g[:, 4])
            try:
                iou = mmcv_box_iou_rotated(b, g, aligned=False).detach().float().clamp_(0, 1)
                return iou.cpu()
            except Exception:
                pass  # fall through to shapely/aabb

        if _SHAPELY_OK:
            return self._iou_shapely(boxes, gts).clamp_(0, 1)
        return self._iou_aabb(boxes, gts).clamp_(0, 1)

    # --------------- shapely / aabb helpers ---------------
    def _to_poly(self, obb_np: np.ndarray) -> List["Polygon"]:
        out = []
        for cx, cy, w, h, ang in obb_np:
            ca, sa = math.cos(float(ang)), math.sin(float(ang))
            dx, dy = w / 2.0, h / 2.0
            pts = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
            R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
            rot = pts @ R.T
            rot[:, 0] += cx; rot[:, 1] += cy
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
                if inter <= 0:
                    continue
                u = pa.area + pb.area - inter
                out[i, j] = float(inter / u) if u > 0 else 0.0
        return torch.from_numpy(out)

    def _iou_aabb(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        def aabb(x):
            cx, cy, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
            return cx - w/2, cy - h/2, cx + w/2, cy + h/2

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
                iw = max(0.0, xx2 - xx1); ih = max(0.0, yy2 - yy1)
                inter = iw * ih
                ua = (float(a3[i]-a1[i]) * float(a4[i]-a2[i]))
                ub = (float(b3[j]-b1[j]) * float(b4[j]-b2[j]))
                union = ua + ub - inter
                out[i, j] = float(inter / union) if union > 0 else 0.0
        return out

    # --------------- metrics ---------------
    def _ap_from_pr(self, tp: np.ndarray, fp: np.ndarray, sc: np.ndarray) -> float:
        if tp.size == 0:
            return 0.0
        order = np.argsort(-sc)
        tp = tp[order].astype(np.float32)
        fp = fp[order].astype(np.float32)
        ctp = np.cumsum(tp); cfp = np.cumsum(fp)
        denom = max(1.0, float(tp.sum()))
        recall = ctp / denom
        precision = ctp / np.maximum(ctp + cfp, 1e-9)
        rec_points = np.linspace(0, 1, 101)
        prec_at_rec = np.zeros_like(rec_points)
        for i, r in enumerate(rec_points):
            inds = np.where(recall >= r)[0]
            if inds.size:
                prec_at_rec[i] = precision[inds].max()
        return float(prec_at_rec.mean())

    def _finalize(self, st: Dict[str, Any]) -> Dict[str, Any]:
        aps = []
        for t in self.iou_thrs:
            tp = np.array(st["tp_by_thr"][float(t)], dtype=np.float32)
            fp = np.array(st["fp_by_thr"][float(t)], dtype=np.float32)
            sc = np.array(st["scores_by_thr"][float(t)], dtype=np.float32)
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
            pck = 0.0; pck_any = 0.0

        return {
            "mAP50": mAP50, "mAP": mAP,
            "pck@0.05": pck, "pck_any@0.05": pck_any,
            "tps": float(st["recall_hits"][0.5]),
            "pred_per_img": pred_per_img,
            "images": int(images),
            "best_iou": np.float64(best_iou),
            "recall@0.1": rec01, "recall@0.3": rec03, "recall@0.5": rec05,
        }

    def _pretty(self, m: Dict[str, Any]):
        logging.getLogger("evaluator_full").info(
            "[EVAL] images %.1f  mAP50 %.6f  mAP %.6f  pck@0.05 %.6f  pck_any@0.05 %.6f  "
            "tps %.1f  pred_per_img %.2f  recall@0.1 %.2f  recall@0.3 %.2f  recall@0.5 %.2f  best_iou %.3f",
            m.get("images", 0.0), m.get("mAP50", 0.0), m.get("mAP", 0.0),
            m.get("pck@0.05", 0.0), m.get("pck_any@0.05", 0.0),
            m.get("tps", 0.0), m.get("pred_per_img", 0.0),
            m.get("recall@0.1", 0.0), m.get("recall@0.3", 0.0), m.get("recall@0.5", 0.0),
            m.get("best_iou", 0.0),
        )