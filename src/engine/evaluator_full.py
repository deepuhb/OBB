# src/engine/evaluator_full.py

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---- optional fast geometry backends ----
try:
    from mmcv.ops import box_iou_rotated as mmcv_box_iou_rotated
    from mmcv.ops import nms_rotated as mmcv_nms_rotated
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
        nms_iou=0.5,
        max_det=100,
        iou_backend=("mmcv" if _MMCV_OK else "shapely"),
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
    d["nms_iou"] = float(d["nms_iou"])
    d["max_det"] = int(d["max_det"])
    d["iou_backend"] = str(d["iou_backend"])
    d["map_iou_st"] = float(d["map_iou_st"])
    d["map_iou_ed"] = float(d["map_iou_ed"])
    d["map_iou_step"] = float(d["map_iou_step"])
    d["pck_alpha"] = float(d["pck_alpha"])
    d["print_every"] = int(d["print_every"])
    return d


class EvaluatorFull:
    """
    OBB + single keypoint evaluator.

    Expects val batch like:
      {"image": (B,3,H,W),
       "bboxes": list[Ti,5]  (cx,cy,w,h,rad) px,
       "labels": list[Ti],   ints,
       "kpts":   list[Ti,2]  (x,y) px, ...}
    """

    def __init__(self,
                 cfg: Optional[Any] = None,
                 names: Optional[List[str]] = None,
                 logger: Optional[logging.Logger] = None,
                 params: Optional[Dict[str, Any]] = None):
        self.log = logger or logging.getLogger("evaluator_full")
        self.names = names or []
        self.cfg = _cfg_to_params(cfg, params)

        if self.cfg["iou_backend"] == "mmcv" and not _MMCV_OK:
            self.log.warning("[eval] iou_backend='mmcv' requested but MMCV ops not found; falling back.")
            self.cfg["iou_backend"] = "shapely"
        if self.cfg["iou_backend"] == "shapely" and not _SHAPELY_OK:
            self.log.warning("[eval] Shapely not found; falling back to coarse AABB IoU.")

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
            det_maps, feats = self._split_outs(outs)

            preds_list = self._decode_preds(model_eval, det_maps, imgs, score_thr=self.cfg["score_thresh"])
            if preds_list is None or len(preds_list) != B:
                if not hasattr(self, "_decode_dbg_once"):
                    self._decode_dbg_once = True
                    self._debug_decode_failure(model_eval, det_maps)
                self.log.error("[eval] Could not decode predictions; returning empty preds.")
                preds_list = [self._empty_pred(device) for _ in range(B)]

            gtb_list = batch.get("bboxes", [torch.zeros((0, 5), device=device) for _ in range(B)])
            gtk_list = batch.get("kpts",   [torch.zeros((0, 2), device=device) for _ in range(B)])

            uv_pred = None
            if hasattr(model_eval, "kpt_from_obbs"):
                try:
                    obb_for_kpt = []
                    for pred in preds_list:
                        bx = pred["boxes"]
                        obb_for_kpt.append(bx[: min(self.cfg["max_det"], bx.shape[0])] if bx.numel() else bx)
                    uv_pred, _ = model_eval.kpt_from_obbs(feats, obb_for_kpt)
                except Exception as e:
                    self.log.debug("[eval] kpt_from_obbs failed: %s", str(e))
                    uv_pred = None

            uv_cursor = 0
            for b in range(B):
                img_seen += 1
                stats["images_eval"] += 1
                gt_boxes = gtb_list[b].to(device)
                gt_kpts  = gtk_list[b].to(device)
                preds    = preds_list[b]

                keep = self._rotated_nms(preds["boxes"], preds["scores"],
                                         iou_thr=self.cfg["nms_iou"],
                                         max_det=self.cfg["max_det"])
                boxes  = preds["boxes"][keep]
                scores = preds["scores"][keep]

                stats["pred_count"] += int(boxes.shape[0])
                stats["gt_total"]   += int(gt_boxes.shape[0])

                iou_mat = self._iou_matrix(boxes, gt_boxes)
                stats["best_iou"].append(float(iou_mat.max().item()) if iou_mat.numel() else 0.0)

                order = torch.argsort(scores, descending=True)
                used = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool)
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

                if uv_pred is not None and gt_kpts.numel():
                    n_here = min(boxes.shape[0], uv_pred.shape[0] - uv_cursor)
                    if n_here > 0:
                        uv_batch = uv_pred[uv_cursor: uv_cursor + n_here]
                        uv_cursor += n_here
                        sizes = torch.sqrt(boxes[:n_here, 2] * boxes[:n_here, 3]).clamp(min=1.0)
                        thr = self.cfg["pck_alpha"] * sizes
                        dists = torch.cdist(uv_batch.float().to(device), gt_kpts.float().to(device))
                        if dists.numel():
                            dmin, _ = dists.min(dim=1)
                            ok = (dmin <= thr)
                            stats["pck_ok"] += int(ok.sum().item())
                            stats["pck_any_ok"] += int((dmin <= thr.max()).sum().item())
                            stats["pck_total"] += int(n_here)

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

    # --------------- OUTS → (det_maps, feats) ---------------

    def _split_outs(self, outs):
        # tuple/list: (det_maps, feats)
        if isinstance(outs, (tuple, list)):
            if len(outs) == 2:
                return outs[0], outs[1]
            if len(outs) == 1:
                return outs[0], None
            return outs, None  # unknown but pass through

        # dict: common keys
        if isinstance(outs, dict):
            feats = outs.get("feats", outs.get("features", None))
            det = outs.get("det", None)
            if det is None:
                det = outs.get("maps", outs.get("pyramids", outs.get("preds", outs)))
            return det, feats

        # raw tensor or other
        return outs, None

    # --------------- decode helpers ---------------

    def _empty_pred(self, device):
        return {"boxes": torch.zeros((0, 5), device=device),
                "scores": torch.zeros((0,), device=device),
                "cls": None}

    def _decode_preds(self, model_eval, det_maps, imgs, score_thr=0.25):
        """Return list[dict] per image with boxes in px/rad, scores, cls."""
        B, _, H, W = imgs.shape
        device = imgs.device

        # 0) If det_maps is a dict, try to normalize into something decodable
        dict_pyr = None
        if isinstance(det_maps, dict):
            # A) if looks like per-image lists already
            if "boxes" in det_maps and isinstance(det_maps["boxes"], list):
                return self._normalize_decoded(det_maps, (W, H), device, score_thr)

            # B) pull (B,C,Hf,Wf) tensors and order by inferred stride
            dict_pyr = self._dict_to_pyramid(det_maps, (H, W), device)

        # 1) Known hooks on the model (accept dict or list)
        hook_names = (
            "export_decode", "decode_obb", "decode_detections", "postprocess",
            "predict", "inference", "forward_export", "decode"
        )
        out = self._try_hooks(model_eval, det_maps, imgs, hook_names)
        parsed = self._normalize_decoded(out, (W, H), device, score_thr) if out is not None else None
        if parsed is not None:
            self._maybe_log_decode_path(f"model.{self._last_called_name}")
            return parsed

        # 2) Hooks on common submodules
        for subname in ("head", "det_head", "yolo"):
            sub = getattr(model_eval, subname, None)
            if sub is None:
                continue
            out = self._try_hooks(sub, det_maps, imgs, hook_names)
            parsed = self._normalize_decoded(out, (W, H), device, score_thr) if out is not None else None
            if parsed is not None:
                self._maybe_log_decode_path(f"model.{subname}.{self._last_called_name}")
                return parsed

        # 3) Pyramid decode if we built one from dict
        if dict_pyr is not None:
            for name in ("decode_obb_from_pyramids", "decode_from_pyramids", "decode_yolo", "export_pyramids"):
                fn = getattr(model_eval, name, None)
                if callable(fn):
                    try:
                        out = fn(dict_pyr, imgs)
                        parsed = self._normalize_decoded(out, (W, H), device, score_thr)
                        if parsed is not None:
                            self._maybe_log_decode_path(f"model.{name}(pyramids-from-dict, imgs)")
                            return parsed
                    except Exception as e:
                        self._dbg(f"{name}(dict_pyr, imgs) failed: {e}")

        # 4) If maps are a pyramid list/tuple, try explicit pyramid decode names
        if isinstance(det_maps, (list, tuple)) and len(det_maps) in (3, 4, 5):
            for name in ("decode_obb_from_pyramids", "decode_from_pyramids", "decode_yolo", "export_pyramids"):
                fn = getattr(model_eval, name, None)
                if callable(fn):
                    try:
                        out = fn(det_maps, imgs)
                        parsed = self._normalize_decoded(out, (W, H), device, score_thr)
                        if parsed is not None:
                            self._maybe_log_decode_path(f"model.{name}(pyramids, imgs)")
                            return parsed
                    except Exception as e:
                        self._dbg(f"{name}(pyramids, imgs) failed: {e}")

        # 5) Dense fallback (B,N,K) — from tensor OR from any dict entry that matches
        candidates = []
        if torch.is_tensor(det_maps):
            candidates.append(det_maps)
        elif isinstance(det_maps, dict):
            for v in det_maps.values():
                if torch.is_tensor(v) and v.dim() == 3 and v.size(-1) >= 6:
                    candidates.append(v)
        for t in candidates:
            try:
                parsed = self._decode_dense(t, (W, H), device, score_thr)
                if parsed is not None:
                    self._maybe_log_decode_path("dense_fallback(B,N,K)")
                    return parsed
            except Exception as e:
                self._dbg(f"dense fallback failed: {e}")

        return None

    def _try_hooks(self, obj, det_maps, imgs, names):
        """Try calling obj.<name> with (det_maps, imgs) → (det_maps,) → (imgs,) → ()."""
        for name in names:
            fn = getattr(obj, name, None)
            if not callable(fn):
                continue
            for args in ((det_maps, imgs), (det_maps,), (imgs,), tuple()):
                try:
                    out = fn(*args)
                    self._last_called_name = name
                    return out
                except TypeError:
                    continue
                except Exception as e:
                    self._dbg(f"{obj.__class__.__name__}.{name}{tuple(a.__class__.__name__ for a in args)} failed: {e}")
                    break
        return None

    def _normalize_decoded(self, decoded, wh: Tuple[int, int], device, score_thr):
        W, H = wh

        # A) list[dict]
        if isinstance(decoded, list) and decoded and isinstance(decoded[0], dict):
            out = []
            for d in decoded:
                boxes = d.get("boxes"); scores = d.get("scores")
                cls = d.get("cls", d.get("labels"))
                if boxes is None or scores is None:
                    return None
                boxes = self._ensure_px_rad(boxes.to(device), (W, H))
                scores = scores.to(device)
                if cls is not None:
                    cls = cls.to(device)
                keep = scores >= float(score_thr)
                out.append({"boxes": boxes[keep], "scores": scores[keep], "cls": (cls[keep] if cls is not None else None)})
            return out

        # B) tuple whose first item is list[dict]
        if isinstance(decoded, (list, tuple)) and decoded:
            first = decoded[0]
            if isinstance(first, list) and first and isinstance(first[0], dict):
                return self._normalize_decoded(first, wh, device, score_thr)

        # C) list[tensor] per image, each [N, >=6]
        if isinstance(decoded, list) and decoded and torch.is_tensor(decoded[0]):
            out = []
            for t in decoded:
                if t.numel() == 0:
                    out.append(self._empty_pred(device)); continue
                if t.dim() == 1: t = t[None, :]
                boxes = self._ensure_px_rad(t[:, :5].to(device), (W, H))
                scores = t[:, 5].to(device)
                cls = t[:, 6].long().to(device) if t.size(1) > 6 else None
                keep = scores >= float(score_thr)
                out.append({"boxes": boxes[keep], "scores": scores[keep], "cls": (cls[keep] if cls is not None else None)})
            return out

        # D) dict of per-image lists/tensors
        if isinstance(decoded, dict) and "boxes" in decoded and isinstance(decoded["boxes"], list):
            out = []
            boxes_list = decoded["boxes"]
            scores_list = decoded.get("scores", [None] * len(boxes_list))
            cls_list = decoded.get("cls", decoded.get("labels", [None] * len(boxes_list)))
            for i in range(len(boxes_list)):
                bx = boxes_list[i]; sc = scores_list[i]
                cl = (cls_list[i] if isinstance(cls_list, list) else None)
                bx = self._ensure_px_rad(bx.to(device), (W, H))
                sc = torch.ones((bx.shape[0],), device=device) if sc is None else sc.to(device)
                cl = (cl.to(device) if cl is not None else None)
                keep = sc >= float(score_thr)
                out.append({"boxes": bx[keep], "scores": sc[keep], "cls": (cl[keep] if cl is not None else None)})
            return out

        # E) dense (B,N,K)
        if torch.is_tensor(decoded) and decoded.dim() == 3 and decoded.size(-1) >= 6:
            return self._decode_dense(decoded, (W, H), device, score_thr)

        return None

    def _decode_dense(self, t: torch.Tensor, wh: Tuple[int, int], device, score_thr):
        B, N, K = t.shape
        W, H = wh
        t = t.to(device)

        boxes = t[..., :5]      # cx,cy,w,h,angle
        scores = t[..., 5]
        cls = t[..., 6].long() if K > 6 else None

        # heuristic: normalized?
        cx_med = boxes[..., 0].detach().abs().median()
        cy_med = boxes[..., 1].detach().abs().median()
        norm_like = (cx_med <= 1.2) and (cy_med <= 1.2)
        if norm_like:
            boxes = boxes.clone()
            boxes[..., 0] *= W; boxes[..., 1] *= H
            boxes[..., 2] *= W; boxes[..., 3] *= H

        # heuristic: degrees?
        ang_med = boxes[..., 4].detach().abs().median()
        if float(ang_med) > 3.5:
            boxes = boxes.clone()
            boxes[..., 4] = torch.deg2rad(boxes[..., 4].clamp(min=-360.0, max=360.0))

        boxes[..., 2] = boxes[..., 2].clamp(min=1.0, max=W * 2.0)
        boxes[..., 3] = boxes[..., 3].clamp(min=1.0, max=H * 2.0)

        outs = []
        for b in range(B):
            s = scores[b]
            k = s >= float(score_thr)
            outs.append({
                "boxes": boxes[b][k],
                "scores": s[k],
                "cls": (cls[b][k] if cls is not None else None)
            })
        return outs

    def _ensure_px_rad(self, boxes: torch.Tensor, wh: Tuple[int, int]):
        if boxes.numel() == 0:
            return boxes
        W, H = wh
        bx = boxes.clone()

        if bx.size(-1) > 5:
            bx = bx[:, :5]

        cx_med = bx[:, 0].detach().abs().median()
        cy_med = bx[:, 1].detach().abs().median()
        if (cx_med <= 1.2) and (cy_med <= 1.2):
            bx[:, 0] *= W; bx[:, 1] *= H
            bx[:, 2] *= W; bx[:, 3] *= H

        ang_med = bx[:, 4].detach().abs().median()
        if float(ang_med) > 3.5:
            bx[:, 4] = torch.deg2rad(bx[:, 4].clamp(min=-360.0, max=360.0))

        bx[:, 2] = bx[:, 2].clamp(min=1.0, max=W * 2.0)
        bx[:, 3] = bx[:, 3].clamp(min=1.0, max=H * 2.0)
        return bx

    # --------------- dict → pyramid ---------------

    def _dict_to_pyramid(self, d: Dict[str, Any], hw: Tuple[int, int], device) -> Optional[List[torch.Tensor]]:
        """Extract (B,C,Hf,Wf) maps from dict and sort by inferred stride."""
        H, W = hw
        items: List[Tuple[float, torch.Tensor]] = []
        for k, v in d.items():
            if torch.is_tensor(v) and v.dim() == 4:
                _, _, Hf, Wf = v.shape
                # inferred stride ~ image_size / feature_size
                sH = max(1.0, H / max(1, Hf))
                sW = max(1.0, W / max(1, Wf))
                s = float((sH + sW) * 0.5)
                items.append((s, v.to(device)))
        if not items:
            return None
        # sort by increasing stride (P3→P4→P5…)
        items.sort(key=lambda x: x[0])
        return [v for _, v in items]

    # --------------- IoU / NMS ---------------

    def _rotated_nms(self, boxes: torch.Tensor, scores: torch.Tensor,
                     iou_thr: float, max_det: int) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        if self.cfg["iou_backend"] == "mmcv" and _MMCV_OK:
            b = boxes.clone()
            b[:, 4] = torch.rad2deg(b[:, 4])
            try:
                res = mmcv_nms_rotated(b, scores, float(iou_thr))
                keep = res[1] if (isinstance(res, (tuple, list)) and len(res) == 2) else res
                return keep[:max_det] if keep.numel() > max_det else keep
            except Exception as e:
                self.log.warning("[eval] mmcv_nms_rotated failed (%s); using greedy fallback.", str(e))
        order = torch.argsort(scores, descending=True)
        keep = []
        sup = torch.zeros_like(order, dtype=torch.bool)
        polys = self._to_poly(boxes.detach().cpu().numpy()) if _SHAPELY_OK else None
        for i_idx in range(order.numel()):
            i = int(order[i_idx])
            if sup[i_idx]:
                continue
            keep.append(i)
            if len(keep) >= max_det:
                break
            for j_idx in range(i_idx + 1, order.numel()):
                if sup[j_idx]:
                    continue
                j = int(order[j_idx])
                if _SHAPELY_OK:
                    pi, pj = polys[i], polys[j]
                    inter = pi.intersection(pj).area
                    u = pi.area + pj.area - inter
                    iou = (inter / u) if u > 0 else 0.0
                else:
                    iou = float(self._iou_aabb(boxes[i:i+1], boxes[j:j+1])[0, 0].item())
                if iou >= iou_thr:
                    sup[j_idx] = True
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def _iou_matrix(self, boxes: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:
        N = int(boxes.shape[0]); M = int(gts.shape[0])
        if N == 0 or M == 0:
            return torch.zeros((N, M))

        if self.cfg["iou_backend"] == "mmcv" and _MMCV_OK:
            b = boxes.clone(); g = gts.clone()
            b[:, 4] = torch.rad2deg(b[:, 4])
            g[:, 4] = torch.rad2deg(g[:, 4])
            dev = boxes.device if boxes.is_cuda else (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            b = b.to(dev); g = g.to(dev)
            iou = mmcv_box_iou_rotated(b, g, aligned=False).detach().float().clamp_(0, 1)
            return iou.cpu()

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

        images = float(st["images_eval"])
        pred_per_img = float(st["pred_count"]) / max(images, 1.0)
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
        self.log.info(
            "[EVAL] images %.1f  mAP50 %.6f  mAP %.6f  pck@0.05 %.6f  pck_any@0.05 %.6f  "
            "tps %.1f  pred_per_img %.2f  recall@0.1 %.2f  recall@0.3 %.2f  recall@0.5 %.2f  best_iou %.3f",
            m.get("images", 0.0), m.get("mAP50", 0.0), m.get("mAP", 0.0),
            m.get("pck@0.05", 0.0), m.get("pck_any@0.05", 0.0),
            m.get("tps", 0.0), m.get("pred_per_img", 0.0),
            m.get("recall@0.1", 0.0), m.get("recall@0.3", 0.0), m.get("recall@0.5", 0.0),
            m.get("best_iou", 0.0),
        )

    # --------------- debug helpers ---------------

    def _maybe_log_decode_path(self, path: str):
        if not hasattr(self, "_dec_path_printed"):
            self._dec_path_printed = True
            self.log.info("[eval] decode path: %s", path)

    def _dbg(self, msg: str):
        if not hasattr(self, "_dbg_printed"):
            self._dbg_printed = 0
        if self._dbg_printed < 6:
            self._dbg_printed += 1
            self.log.debug("[eval] %s", msg)

    def _debug_decode_failure(self, model_eval, det_maps):
        cand = []
        for n in dir(model_eval):
            if any(k in n for k in ("decode", "post", "export", "predict")):
                try:
                    if callable(getattr(model_eval, n)):
                        cand.append(n)
                except Exception:
                    pass
        if isinstance(det_maps, dict):
            shapes = {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in det_maps.items()}
            keys = ", ".join(list(det_maps.keys()))
            self.log.warning(
                "[eval] decode failed. Candidate methods on model: %s | det_maps=dict keys=[%s] shapes=%s",
                (", ".join(sorted(cand)) or "<none>"), keys, shapes
            )
        else:
            shape_str = (f"tensor{tuple(det_maps.shape)}" if torch.is_tensor(det_maps)
                         else f"pyramid[{len(det_maps)}]" if isinstance(det_maps, (list, tuple))
                         else type(det_maps).__name__)
            self.log.warning(
                "[eval] decode failed. Candidate methods on model: %s | det_maps=%s",
                (", ".join(sorted(cand)) or "<none>"), shape_str
            )
