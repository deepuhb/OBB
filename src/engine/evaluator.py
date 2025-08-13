# src/engine/evaluator.py
import torch
import torchvision
import numpy as np
from ..models.postprocess.decode import decode

class Evaluator:
    """
    Computes single-threshold AP (mAP@IoU_thr) and PCK@tau over matched pairs.
    """
    def __init__(self, cfg=None, debug=False):
        cfg = cfg or {}
        ev = cfg.get("eval", {})
        self.score_thresh = float(ev.get("score_thresh", 0.10))  # align with decoder
        self.nms_iou      = float(ev.get("nms_iou", 0.50))
        self.iou_thr      = float(ev.get("iou_thr", 0.50))
        self.pck_tau      = float(ev.get("pck_tau", 0.05))
        self.max_det      = int(ev.get("max_det", 100))
        self.strides      = tuple(cfg.get("model", {}).get("strides", (4, 8, 16)))
        self.debug        = bool(debug)

    @torch.no_grad()
    def evaluate(self, model, loader, device="cpu"):
        model.eval()
        all_scores = []
        all_matches = []
        total_gt = 0

        pck_hits = 0
        pck_total = 0
        images_count = 0

        for batch in loader:
            images_count += len(batch["image"])
            imgs = batch["image"].to(device)
            outs = model(imgs)

            preds = decode(
                outs["det"], outs["kpt"],
                strides=self.strides,
                score_thresh=self.score_thresh,
                iou_thresh=self.nms_iou,
                num_classes=1,
                max_det=self.max_det,
            )

            for i in range(len(imgs)):
                gt_boxes = batch["boxes"][i].to(device) if len(batch["boxes"][i]) else torch.zeros((0,4), device=device)
                gt_kpts  = batch["kpts"][i].to(device)  if len(batch["kpts"][i])  else torch.zeros((0,2), device=device)
                total_gt += int(gt_boxes.shape[0])

                pr = preds[i]
                pb = pr["boxes"]; ps = pr["scores"]; pk = pr["kpts"]

                if self.debug:
                    ngt = int(gt_boxes.shape[0])
                    npr = int(pb.shape[0])
                    msg = f"[eval] img#{i}: GT={ngt}, pred={npr}"
                    if npr:
                        msg += f", score[min/mean/max]=[{ps.min().item():.3f}/{ps.mean().item():.3f}/{ps.max().item():.3f}]"
                    print(msg)

                if pb.numel() == 0:
                    # no predictions for this image
                    continue

                # Sort predictions by score desc
                order = torch.argsort(ps, descending=True)
                pb = pb[order]; pk = pk[order]; ps = ps[order]

                if gt_boxes.numel() > 0:
                    ious = torchvision.ops.box_iou(pb, gt_boxes)  # (P,G)
                else:
                    ious = torch.zeros((pb.shape[0], 0), device=device)

                taken = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=device)
                for pidx in range(pb.shape[0]):
                    all_scores.append(float(ps[pidx]))
                    if gt_boxes.numel() == 0:
                        all_matches.append(0)
                        continue
                    i_best = torch.argmax(ious[pidx])
                    if ious[pidx, i_best] >= self.iou_thr and not taken[i_best]:
                        all_matches.append(1)
                        taken[i_best] = True
                        # PCK on matched pair
                        w = (gt_boxes[i_best, 2] - gt_boxes[i_best, 0]).clamp_min(1.0)
                        h = (gt_boxes[i_best, 3] - gt_boxes[i_best, 1]).clamp_min(1.0)
                        thr = self.pck_tau * torch.maximum(w, h)
                        d = torch.linalg.vector_norm(pk[pidx] - gt_kpts[i_best])
                        pck_hits += int(d <= thr)
                        pck_total += 1
                    else:
                        all_matches.append(0)

        # ---- AP computation (11-point interpolation) ----
        if len(all_scores) == 0 or total_gt == 0:
            map50 = 0.0
        else:
            scores = np.array(all_scores, dtype=np.float64)
            matches = np.array(all_matches, dtype=np.int32)
            order = np.argsort(-scores)
            matches = matches[order]
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

        pck = float(pck_hits / max(1, pck_total))
        return {"map50": map50, "pck@0.05": pck, "images": images_count}
