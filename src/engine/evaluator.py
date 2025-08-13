# src/engine/evaluator.py
import torch
import torchvision
import numpy as np
from types import SimpleNamespace
from ..models.postprocess.decode import decode

def _ns(x):
    return x if isinstance(x, SimpleNamespace) else SimpleNamespace(**(x or {}))

def _get(ns, key, default):
    """Safe getter: returns default if ns is None or value is None."""
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

class Evaluator:
    def __init__(self, cfg, debug=False):
        self.cfg = cfg
        ev = _ns(getattr(cfg, "eval", None))
        md = _ns(getattr(cfg, "model", None))
        dc = _ns(getattr(cfg, "decode", None))

        # Eval knobs (loose early; you can tighten later)
        self.score_thresh = float(_get(ev, "score_thresh", 0.0))
        self.iou_thr      = float(_get(ev, "iou_thr", 0.3))
        self.pck_tau      = float(_get(ev, "pck_tau", 0.10))
        self.max_det      = int(_get(ev, "max_det", 5000))
        self.topk         = int(_get(ev, "topk", 10000))
        self.rank_by      = str(_get(ev, "rank_by", "score"))
        self.strides      = tuple(_get(md, "strides", (4, 8, 16)))
        self.debug        = bool(debug)

        # Decode args (used inside evaluate)
        self.dec_args = dict(
            strides=self.strides,
            score_thresh=float(_get(dc, "score_thresh", 0.0)),
            nms_iou=float(_get(dc, "nms_iou", 0.50)),
            max_det=int(_get(dc, "max_det", 5000)),
            topk=int(_get(dc, "topk", 10000)),
            apply_nms=_to_bool(_get(dc, "apply_nms", False)),
        )

        if self.debug:
            print("[evaluator] eval:",
                  dict(score_thresh=self.score_thresh, iou_thr=self.iou_thr,
                       pck_tau=self.pck_tau, max_det=self.max_det, topk=self.topk,
                       rank_by=self.rank_by))
            print("[evaluator] decode:", self.dec_args)


    @torch.no_grad()
    def evaluate(self, model, loader, device="cpu"):
        model.eval()
        all_ranks, all_matches = [], []
        total_gt = 0
        pck_hits = pck_total = 0
        pck_any_hits = pck_any_total = 0
        best_iou_sum = 0.0
        best_iou_cnt = 0
        recall_iou1 = recall_iou3 = recall_iou5 = 0
        images_count = 0
        tp_count = 0
        pred_total = 0

        for batch in loader:
            images_count += len(batch["image"])
            imgs = batch["image"].to(device)
            outs = model(imgs)
            preds = decode(outs["det"], outs["kpt"], **self.dec_args)

            for i in range(len(imgs)):
                gt_boxes = batch["boxes"][i].to(device) if len(batch["boxes"][i]) else torch.zeros((0,4), device=device)
                gt_kpts  = batch["kpts"][i].to(device)  if len(batch["kpts"][i])  else torch.zeros((0,2), device=device)
                total_gt += int(gt_boxes.shape[0])

                pr = preds[i]
                pb, ps, pk = pr["boxes"], pr["scores"], pr["kpts"]
                pred_total += int(pb.shape[0])

                # choose ranker
                if pb.numel():
                    if self.rank_by == "iou" and gt_boxes.numel():
                        ious_tmp = torchvision.ops.box_iou(pb, gt_boxes)
                        ranker = ious_tmp.max(dim=1).values
                    else:
                        ranker = ps
                    order = torch.argsort(ranker, descending=True)
                    pb, ps, pk = pb[order], ps[order], pk[order]
                    ranker = ranker[order]
                else:
                    ranker = torch.zeros((0,), device=device)

                if self.debug:
                    s = "N/A" if ps.numel()==0 else f"[{ps.min().item():.3f}/{ps.mean().item():.3f}/{ps.max().item():.3f}]"
                    print(f"[eval] img#{i}: GT={int(gt_boxes.shape[0])}, pred={int(pb.shape[0])}, scores={s}")

                # PCK_any (ignores IoU)
                if gt_kpts.numel() and pk.numel():
                    d = torch.cdist(pk, gt_kpts)
                    best = d.min(dim=0).values
                    wh = (gt_boxes[:,2:] - gt_boxes[:,:2]).clamp_min(1.0)
                    thr = self.pck_tau * torch.maximum(wh[:,0], wh[:,1])
                    pck_any_hits += int((best <= thr).sum().item())
                    pck_any_total += int(gt_kpts.shape[0])

                # Best-IoU probes
                if gt_boxes.numel() and pb.numel():
                    ious_full = torchvision.ops.box_iou(pb, gt_boxes)
                    best_per_gt = ious_full.max(dim=0).values
                    bi = best_per_gt.detach()
                    best_iou_sum += float(bi.sum().item())
                    best_iou_cnt += int(bi.numel())
                    recall_iou1 += int((bi >= 0.1).sum().item())
                    recall_iou3 += int((bi >= 0.3).sum().item())
                    recall_iou5 += int((bi >= 0.5).sum().item())

                # AP + PCK on matched pairs
                if pb.numel() == 0 or gt_boxes.numel() == 0:
                    if ranker.numel():
                        all_ranks.extend(ranker.detach().tolist())
                        all_matches.extend([0]*int(ranker.numel()))
                    continue

                ious = torchvision.ops.box_iou(pb, gt_boxes)
                taken = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=device)
                for pidx in range(pb.shape[0]):
                    all_ranks.append(float(ranker[pidx]))
                    i_best = torch.argmax(ious[pidx])
                    if ious[pidx, i_best] >= self.iou_thr and not taken[i_best]:
                        all_matches.append(1)
                        taken[i_best] = True
                        tp_count += 1
                        w = (gt_boxes[i_best,2]-gt_boxes[i_best,0]).clamp_min(1.0)
                        h = (gt_boxes[i_best,3]-gt_boxes[i_best,1]).clamp_min(1.0)
                        thr = self.pck_tau * torch.maximum(w,h)
                        dist = torch.linalg.vector_norm(pk[pidx] - gt_kpts[i_best])
                        pck_hits += int(dist <= thr)
                        pck_total += 1
                    else:
                        all_matches.append(0)

        # AP (11-point)
        if len(all_ranks) == 0 or total_gt == 0:
            map50 = 0.0
        else:
            ranks  = np.asarray(all_ranks, dtype=np.float64)
            matches= np.asarray(all_matches, dtype=np.int32)
            order  = np.argsort(-ranks)
            matches= matches[order]
            tp = matches
            fp = 1 - matches
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            denom = np.maximum(tp_cum + fp_cum, 1)
            prec = tp_cum / denom
            rec  = tp_cum / max(1, total_gt)

            ap = 0.0
            for r in np.linspace(0, 1, 11):
                p = prec[rec >= r].max() if np.any(rec >= r) else 0.0
                ap += p / 11.0
            map50 = float(ap)

        best_iou_mean = (best_iou_sum / max(1, best_iou_cnt))
        R = total_gt if total_gt > 0 else 1
        return {
            "map50": map50,
            "pck@0.05": float(pck_hits / max(1, pck_total)),
            "pck_any@0.05": float(pck_any_hits / max(1, pck_any_total)),
            "images": images_count,
            "tp_count": int(tp_count),
            "pred_per_img_avg": float(pred_total / max(1, images_count)),
            "recall@0.1": float(recall_iou1 / R),
            "recall@0.3": float(recall_iou3 / R),
            "recall@0.5": float(recall_iou5 / R),
            "best_iou_mean": float(best_iou_mean),
        }
