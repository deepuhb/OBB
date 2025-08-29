
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class Evaluator:
    """
    Minimal evaluator for OBB detection that:
      • runs ONLY on rank-0,
      • uses the model's decode_obb_from_pyramids(det_maps, imgs, ...),
      • computes mAP over IoU thresholds 0.50:0.95:0.05,
      • uses MMCV's rotated IoU (angles expected in **radians** from the model, converted to degrees).
    """

    def __init__(
        self,
        cfg: Optional[Any] = None,
        names: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log = logger or logging.getLogger("evaluator")
        self.names = names or []

        # Decoder & AP grid defaults
        self.cfg: Dict[str, Any] = dict(
            conf_thres=0.25,
            iou_thres=0.70,   # decoder/NMS IoU
            multi_label=False,
            use_nms=True,
            max_det=100,
            map_iou_st=0.50,
            map_iou_ed=0.95,
            map_iou_step=0.05,
            per_image_debug=3,  # number of images to print mini-stats for
        )

        def _merge(d: Dict[str, Any], src: Any):
            if not src:
                return
            if isinstance(src, dict):
                for k, v in src.items():
                    if k in d:
                        d[k] = v
            else:
                for k in dir(src):
                    if k.startswith("_"):
                        continue
                    try:
                        v = getattr(src, k)
                    except Exception:
                        continue
                    if not callable(v) and k in d:
                        d[k] = v

        if cfg is not None:
            _merge(self.cfg, getattr(cfg, "evaluator", cfg))
        if params:
            _merge(self.cfg, params)

        self.iou_thrs = np.arange(
            float(self.cfg["map_iou_st"]), float(self.cfg["map_iou_ed"]) + 1e-9, float(self.cfg["map_iou_step"])
        ).round(2)

        self._printed_images = 0

    @torch.no_grad()
    def evaluate(
        self, model: torch.nn.Module, val_loader, device: torch.device, max_images: Optional[int] = None
    ) -> Dict[str, float]:

        # Only rank-0 evaluates
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return {"mAP50": 0.0, "mAP": 0.0, "pred_per_img": 0.0, "images": 0}

        # Lazy import MMCV **here** (after CUDA device is set) to avoid init issues
        try:
            from mmcv.ops import box_iou_rotated as mmcv_box_iou_rotated  # type: ignore
            print("[EVAL] IoU backend: mmcv.ops.box_iou_rotated")
        except Exception as e:
            raise RuntimeError(
                f"MMCV rotated IoU is required but could not be imported: {e}"
            )

        model_eval = model.module if hasattr(model, "module") else model
        model_eval.eval()

        stats = {
            "tp_by_thr": {float(t): [] for t in self.iou_thrs},
            "fp_by_thr": {float(t): [] for t in self.iou_thrs},
            "scores_by_thr": {float(t): [] for t in self.iou_thrs},
            "gt_total": 0, "pred_total": 0, "images_eval": 0,
        }

        img_seen = 0
        for batch in val_loader:
            if max_images is not None and img_seen >= max_images:
                break
            if not isinstance(batch, dict) or ("image" not in batch):
                continue

            imgs = batch["image"].to(device, non_blocking=True).float()
            B = int(imgs.shape[0])

            # Forward
            outs = model_eval(imgs)
            if isinstance(outs, (list, tuple)):
                det_maps = outs[0]
            elif isinstance(outs, dict):
                det_maps = outs.get("det")
            else:
                det_maps = outs

            # Decode with imgs
            try:
                preds_list = (model.module if hasattr(model, "module") else model).decode_obb_from_pyramids(
                    det_maps,
                    imgs,
                    score_thr=float(self.cfg["conf_thres"]),
                    iou_thres=float(self.cfg["iou_thres"]),
                    max_det=int(self.cfg["max_det"]),
                    use_nms=bool(self.cfg["use_nms"]),
                    multi_label=bool(self.cfg["multi_label"]),
                )
            except Exception as e:
                self.log.error("[eval] decode failed: %s", str(e))
                preds_list = None

            if not isinstance(preds_list, list) or len(preds_list) != B:
                preds_list = [
                    {"boxes": torch.zeros((0, 5), device=device),
                     "scores": torch.zeros((0,), device=device),
                     "labels": torch.zeros((0,), dtype=torch.long, device=device)}
                    for _ in range(B)
                ]

            gtb_list = batch.get("bboxes", [torch.zeros((0, 5), device=device) for _ in range(B)])
            gtl_list = batch.get("labels", [None for _ in range(B)])

            K = int(self.cfg["per_image_debug"])
            for b in range(B):
                preds = preds_list[b]
                p_boxes = preds["boxes"].to(device).float()      # (N,5) cx,cy,w,h,ang(rad)
                p_scores = preds["scores"].to(device).float().view(-1)
                p_labels = preds.get("labels")
                p_labels = (torch.as_tensor(p_labels, device=device, dtype=torch.long).view(-1)
                            if p_labels is not None else None)

                g_boxes = gtb_list[b].to(device).float()
                g_labels = (torch.as_tensor(gtl_list[b], device=device, dtype=torch.long).view(-1)
                            if gtl_list[b] is not None else None)

                stats["pred_total"] += int(p_boxes.shape[0])
                stats["gt_total"] += int(g_boxes.shape[0])

                # mmcv expects degrees
                if p_boxes.numel() and g_boxes.numel():
                    pb = p_boxes.clone()
                    gb = g_boxes.clone()
                    pb[:, 4] = torch.rad2deg(pb[:, 4])
                    gb[:, 4] = torch.rad2deg(gb[:, 4])
                    iou_mat = mmcv_box_iou_rotated(pb, gb, aligned=False).detach().float().clamp_(0, 1)
                else:
                    iou_mat = torch.zeros((p_boxes.shape[0], g_boxes.shape[0]), device=device)

                # Optional per-image print
                if self._printed_images < K:
                    mn = float(p_scores.min().item()) if p_scores.numel() else 0.0
                    me = float(p_scores.mean().item()) if p_scores.numel() else 0.0
                    mx = float(p_scores.max().item()) if p_scores.numel() else 0.0
                    ok = 0
                    if iou_mat.numel() and g_boxes.numel():
                        ok = int((iou_mat.max(dim=0).values >= float(self.cfg["iou_thres"])).sum().item())
                    print(f"[EVAL IMG {img_seen + b}] pred={int(p_boxes.shape[0])} gt={int(g_boxes.shape[0])}\n"
                          f"  recall@{float(self.cfg['iou_thres']):.2f} = {ok}/{int(g_boxes.shape[0])}  "
                          f"score[min/mean/max]=[{mn:.3f}/{me:.3f}/{mx:.3f}]")
                    self._printed_images += 1

                # Greedy matching per threshold
                order = torch.argsort(p_scores, descending=True)

                for thr in self.iou_thrs:
                    t = float(thr)
                    used = torch.zeros((g_boxes.shape[0],), dtype=torch.bool, device=device)
                    tp, fp = [], []
                    for i in order.tolist():
                        if g_boxes.shape[0] == 0:
                            tp.append(0); fp.append(1); continue

                        # Class-aware matching if labels exist
                        if (p_labels is not None) and (g_labels is not None) and int(g_labels.numel()) > 0:
                            li = p_labels[i].item()
                            cand = (g_labels == li).nonzero(as_tuple=True)[0]
                            if cand.numel() == 0:
                                tp.append(0); fp.append(1); continue
                            ious = iou_mat[i, cand] if iou_mat.numel() else torch.zeros_like(cand, dtype=torch.float32)
                            j_rel = int(torch.argmax(ious).item()) if ious.numel() else 0
                            j = int(cand[j_rel].item()) if cand.numel() else 0
                            iou_ij = float(ious[j_rel].item()) if ious.numel() else 0.0
                        else:
                            if iou_mat.numel():
                                j = int(torch.argmax(iou_mat[i]).item())
                                iou_ij = float(iou_mat[i, j].item())
                            else:
                                j, iou_ij = 0, 0.0

                        if (iou_ij >= t) and (not bool(used[j].item())):
                            tp.append(1); fp.append(0); used[j] = True
                        else:
                            fp.append(1); tp.append(0)

                    stats["tp_by_thr"][t].extend(tp)
                    stats["fp_by_thr"][t].extend(fp)
                    stats["scores_by_thr"][t].extend([float(s) for s in p_scores[order].tolist()])

            img_seen += B
            stats["images_eval"] += B

        # Aggregate mAP
        aps = []
        for t in self.iou_thrs:
            t = float(t)
            tp = np.asarray(stats["tp_by_thr"][t], dtype=np.float32)
            fp = np.asarray(stats["fp_by_thr"][t], dtype=np.float32)
            sc = np.asarray(stats["scores_by_thr"][t], dtype=np.float32)
            aps.append(self._ap_from_pr(tp, fp, sc))
        mAP = float(np.mean(aps)) if aps else 0.0
        try:
            i50 = list(self.iou_thrs).index(0.5)
            mAP50 = float(aps[i50]) if aps else 0.0
        except ValueError:
            mAP50 = 0.0

        images = float(stats["images_eval"]) or 1.0
        pred_per_img = float(stats["pred_total"]) / images
        metrics = {"mAP50": mAP50, "mAP": mAP, "pred_per_img": pred_per_img, "images": int(images)}
        print(f"[EVAL] images {metrics['images']}  mAP50 {metrics['mAP50']:.6f}  mAP {metrics['mAP']:.6f}  "
              f"pred/img {metrics['pred_per_img']:.2f}")
        return metrics

    # --- helpers ----------------------------------------------------------------
    def _ap_from_pr(self, tp: np.ndarray, fp: np.ndarray, sc: np.ndarray) -> float:
        if tp.size == 0:
            return 0.0
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
            if inds.size:
                prec_at_rec[i] = precision[inds].max()
        return float(prec_at_rec.mean())
