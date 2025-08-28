
# evaluator.py (DFL-only OBB evaluator with rotated IoU/NMS support)
from typing import Any, Dict, List, Optional, Tuple
import math
import time
import numpy as np
import torch
import torch.nn.functional as F

try:
    from mmcv.ops import nms_rotated as mmcv_nms_rotated  # type: ignore
    from mmcv.ops import box_iou_rotated as mmcv_box_iou_rotated  # type: ignore
    _HAS_MMCV = True
except Exception:
    mmcv_nms_rotated = None
    mmcv_box_iou_rotated = None
    _HAS_MMCV = False


def _to_cpu_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def _rotated_box_to_poly_xy(box: np.ndarray) -> np.ndarray:
    xc, yc, w, h, ang = box.tolist()
    ca, sa = math.cos(ang), math.sin(ang)
    hw, hh = w * 0.5, h * 0.5
    pts = np.array([[-hw, -hh], [ hw, -hh], [ hw,  hh], [-hw,  hh]], dtype=np.float32)
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)
    rot = pts @ R.T
    rot[:, 0] += xc
    rot[:, 1] += yc
    return rot


def _aabb_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1 = a[:, 0].min(), a[:, 1].min()
    ax2, ay2 = a[:, 0].max(), a[:, 1].max()
    bx1, by1 = b[:, 0].min(), b[:, 1].min()
    bx2, by2 = b[:, 0].max(), b[:, 1].max()
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    denom = area_a + area_b - inter + 1e-9
    return float(inter / denom) if denom > 0 else 0.0


def _rotated_iou_numpy(a_boxes: np.ndarray, b_boxes: np.ndarray) -> np.ndarray:
    if a_boxes.size == 0 or b_boxes.size == 0:
        return np.zeros((a_boxes.shape[0], b_boxes.shape[0]), dtype=np.float32)
    if _HAS_MMCV:
        a_t = torch.tensor(a_boxes, dtype=torch.float32)
        b_t = torch.tensor(b_boxes, dtype=torch.float32)
        a_deg = a_t.clone()
        b_deg = b_t.clone()
        a_deg[:, 4] = a_deg[:, 4] * (180.0 / math.pi)
        b_deg[:, 4] = b_deg[:, 4] * (180.0 / math.pi)
        iou = mmcv_box_iou_rotated(a_deg, b_deg, aligned=False)
        return _to_cpu_numpy(iou)
    polys_a = np.stack([_rotated_box_to_poly_xy(b) for b in a_boxes], axis=0)
    polys_b = np.stack([_rotated_box_to_poly_xy(b) for b in b_boxes], axis=0)
    Na, Nb = polys_a.shape[0], polys_b.shape[0]
    out = np.zeros((Na, Nb), dtype=np.float32)
    for i in range(Na):
        for j in range(Nb):
            out[i, j] = _aabb_iou_xyxy(polys_a[i], polys_b[j])
    return out


def _nms_rotated(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    if _HAS_MMCV:
        det = boxes.clone()
        det[:, 4] = det[:, 4] * (180.0 / math.pi)
        keep, _ = mmcv_nms_rotated(det, scores, iou_threshold=float(iou_thr))
        return keep.to(torch.long)

    b_np = _to_cpu_numpy(boxes)
    s_np = _to_cpu_numpy(scores)
    order = s_np.argsort()[::-1]
    keep_idx = []
    while order.size > 0:
        i = order[0]
        keep_idx.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _rotated_iou_numpy(b_np[i:i+1], b_np[rest])[0]
        rest = rest[ious <= iou_thr]
        order = rest
    return torch.tensor(keep_idx, dtype=torch.long, device=boxes.device)


def _compat_decode_call(head: Any,
                        pyramids: List[torch.Tensor],
                        conf: float,
                        iou: float,
                        max_det: int,
                        use_nms: bool) -> Dict[str, torch.Tensor]:
    kw_try = [
        dict(conf_thresh=conf, iou=iou, max_det=max_det, use_nms=use_nms),
        dict(conf_thres=conf, iou=iou, max_det=max_det, use_nms=use_nms),
        dict(score_thr=conf, iou_thr=iou, max_det=max_det, use_nms=use_nms),
        dict(conf_thresh=conf, iou_thr=iou, max_det=max_det, use_nms=use_nms),
    ]
    for kw in kw_try:
        try:
            out = head.decode_obb_from_pyramids(pyramids, **kw)
            if isinstance(out, dict) and "boxes" in out and "scores" in out:
                return out
        except TypeError:
            continue
        except Exception as e:
            raise
    return _local_decode_obb_from_pyramids(head, pyramids, conf, iou, max_det, use_nms)


def _local_decode_obb_from_pyramids(head: Any,
                                    pyramids: List[torch.Tensor],
                                    conf: float,
                                    iou: float,
                                    max_det: int,
                                    use_nms: bool) -> Dict[str, torch.Tensor]:
    assert hasattr(head, "num_classes"), "head must expose num_classes"
    assert hasattr(head, "reg_max"), "head must expose reg_max"
    assert hasattr(head, "strides"), "head must expose strides"

    C = int(head.num_classes)
    R = int(head.reg_max) + 1
    device = pyramids[0].device

    boxes_all = []
    scores_all = []
    labels_all = []

    for lvl, pm in enumerate(pyramids):
        B, Ch, H, W = pm.shape
        stride = int(head.strides[lvl])
        expect_ch = 1 + C + 1 + 1 + R + R + 1
        if Ch < expect_ch:
            raise RuntimeError(f"Local decoder expects >= {expect_ch} channels, got {Ch} at level {lvl}")

        i = 0
        obj = pm[:, i:i+1]; i += 1
        cls = pm[:, i:i+C]; i += C
        tx = pm[:, i:i+1]; i += 1
        ty = pm[:, i:i+1]; i += 1
        w_logits = pm[:, i:i+R]; i += R
        h_logits = pm[:, i:i+R]; i += R
        angle = pm[:, i:i+1]; i += 1

        S = H * W
        obj = obj.flatten(2).transpose(1, 2)
        cls = cls.flatten(2).transpose(1, 2)
        tx = tx.flatten(2).transpose(1, 2)
        ty = ty.flatten(2).transpose(1, 2)
        w_logits = w_logits.flatten(2).transpose(1, 2)
        h_logits = h_logits.flatten(2).transpose(1, 2)
        angle = angle.flatten(2).transpose(1, 2)

        obj_p = obj.sigmoid()
        cls_p = cls.sigmoid()
        score = (obj_p * cls_p.max(dim=-1, keepdim=True).values).squeeze(-1)

        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        grid_x = grid_x.reshape(1, S, 1).repeat(B, 1, 1)
        grid_y = grid_y.reshape(1, S, 1).repeat(B, 1, 1)
        cx = (grid_x + tx).squeeze(-1) * stride
        cy = (grid_y + ty).squeeze(-1) * stride

        pw = torch.softmax(w_logits, dim=-1)
        ph = torch.softmax(h_logits, dim=-1)
        bins = torch.arange(R, device=device, dtype=pw.dtype).view(1, 1, R)
        w_px = ((pw * bins).sum(dim=-1) + 1.0) * (stride / R)
        h_px = ((ph * bins).sum(dim=-1) + 1.0) * (stride / R)

        a = angle.sigmoid().squeeze(-1) * math.pi - (math.pi / 4.0)

        mask = score >= conf
        if mask.any():
            b_idx, s_idx = torch.where(mask)
            sel = score[b_idx, s_idx]
            cls_id = cls_p[b_idx, s_idx].argmax(dim=-1)

            cxg = cx[b_idx, s_idx]
            cyg = cy[b_idx, s_idx]
            wg = w_px[b_idx, s_idx]
            hg = h_px[b_idx, s_idx]
            ag = a[b_idx, s_idx]

            boxes_all.append(torch.stack([cxg, cyg, wg, hg, ag], dim=1))
            scores_all.append(sel)
            labels_all.append(cls_id)

    if len(boxes_all) == 0:
        return dict(boxes=torch.empty(0, 5), scores=torch.empty(0), labels=torch.empty(0, dtype=torch.long))

    boxes = torch.cat(boxes_all, dim=0)
    scores = torch.cat(scores_all, dim=0)
    labels = torch.cat(labels_all, dim=0)

    if use_nms and boxes.numel() > 0:
        keep_all = []
        for c in labels.unique():
            m = labels == c
            if m.sum() == 0:
                continue
            keep_c = _nms_rotated(boxes[m], scores[m], iou_thr=iou)
            if keep_c.numel():
                idx_c = torch.where(m)[0][keep_c]
                keep_all.append(idx_c)
        if len(keep_all):
            keep = torch.cat(keep_all, dim=0)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    if boxes.shape[0] > max_det:
        topk = torch.topk(scores, k=max_det, largest=True).indices
        boxes, scores, labels = boxes[topk], scores[topk], labels[topk]

    return dict(boxes=boxes, scores=scores, labels=labels)


def _parse_gt_for_image(batch: Dict[str, Any], image_index: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    for bx_key in ["gt_boxes", "bboxes", "boxes"]:
        if bx_key in batch and isinstance(batch[bx_key], (list, tuple)) and len(batch[bx_key]) > image_index:
            gtb = batch[bx_key][image_index]
            gtb = gtb.to(device)
            if gtb.numel() and gtb.shape[-1] == 4:
                pad = torch.zeros((gtb.shape[0], 1), dtype=gtb.dtype, device=device)
                gtb = torch.cat([gtb, pad], dim=1)
            gtc = None
            for lc_key in ["gt_labels", "labels", "classes"]:
                if lc_key in batch and isinstance(batch[lc_key], (list, tuple)) and len(batch[lc_key]) > image_index:
                    gtc = batch[lc_key][image_index].to(device).long()
                    break
            if gtc is None:
                gtc = torch.zeros((gtb.shape[0],), dtype=torch.long, device=device)
            return gtb, gtc

    if "targets" in batch and isinstance(batch["targets"], torch.Tensor):
        tgt = batch["targets"]
        if tgt.numel():
            if tgt.shape[-1] >= 6:
                sel = tgt[:, 0] == image_index if tgt.shape[-1] >= 7 else tgt[:, 0] == image_index
                rows = tgt[sel]
                if rows.numel():
                    cls_col = 1
                    xywh_start = 2
                    xc, yc, w, h = rows[:, xywh_start:xywh_start+4].T
                    if rows.shape[1] >= 7:
                        ang = rows[:, xywh_start+4]
                    else:
                        ang = torch.zeros_like(xc)
                    gtb = torch.stack([xc, yc, w, h, ang], dim=1).to(device)
                    gtc = rows[:, cls_col].to(device).long()
                    return gtb, gtc

    return torch.empty(0, 5, device=device), torch.empty(0, dtype=torch.long, device=device)


def _compute_ap50_per_class(all_preds: List[Dict[str, np.ndarray]],
                            all_gts: List[Dict[str, np.ndarray]],
                            num_classes: int,
                            iou_thresh: float = 0.50) -> float:
    aps = []
    for c in range(num_classes):
        dets = []
        npos = 0
        for img_idx, (pred_i, gt_i) in enumerate(zip(all_preds, all_gts)):
            gt_cls_mask = (gt_i["labels"] == c)
            npos += int(gt_cls_mask.sum())
            keep = (pred_i["labels"] == c)
            if keep.sum() == 0:
                continue
            boxes = pred_i["boxes"][keep]
            scores = pred_i["scores"][keep]
            for b, s in zip(boxes, scores):
                dets.append((img_idx, float(s), b))
        if npos == 0:
            continue
        if len(dets) == 0:
            aps.append(0.0)
            continue
        dets.sort(key=lambda x: x[1], reverse=True)
        tp = np.zeros(len(dets), dtype=np.float32)
        fp = np.zeros(len(dets), dtype=np.float32)
        gt_matched = [np.zeros((all_gts[i]["boxes"].shape[0],), dtype=bool) for i in range(len(all_gts))]
        for k, (img_idx, score, box) in enumerate(dets):
            gtb = all_gts[img_idx]["boxes"]
            gtc = all_gts[img_idx]["labels"]
            cand = np.where(gtc == c)[0]
            if cand.size == 0:
                fp[k] = 1.0
                continue
            ious = _rotated_iou_numpy(box.reshape(1, 5), gtb[cand]).reshape(-1)
            if ious.size == 0:
                fp[k] = 1.0
                continue
            j = int(ious.argmax())
            best_iou = float(ious[j])
            j_gt = cand[j]
            if best_iou >= iou_thresh and not gt_matched[img_idx][j_gt]:
                tp[k] = 1.0
                gt_matched[img_idx][j_gt] = True
            else:
                fp[k] = 1.0
        fp_cum = np.cumsum(fp)
        tp_cum = np.cumsum(tp)
        rec = tp_cum / (npos + 1e-9)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
            ap += p / 11.0
        aps.append(ap)
    return float(np.mean(aps)) if len(aps) else 0.0


class Evaluator:
    def __init__(self,
                 iou_thresh: float = 0.50,
                 score_thresh: float = 0.25,
                 use_nms: bool = True,
                 max_det: int = 300,
                 print_debug: bool = True):
        self.iou_thresh = float(iou_thresh)
        self.score_thresh = float(score_thresh)
        self.use_nms = bool(use_nms)
        self.max_det = int(max_det)
        self.print_debug = bool(print_debug)

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, loader, epoch: Optional[int] = None) -> float:
        model.eval()
        device = next(model.parameters()).device
        all_preds_np: List[Dict[str, np.ndarray]] = []
        all_gts_np: List[Dict[str, np.ndarray]] = []
        t0 = time.time()
        n_pred_total = 0
        num_classes = getattr(getattr(model, "head", model), "num_classes", 1)

        for batch_idx, batch in enumerate(loader):
            imgs = batch["imgs"] if isinstance(batch, dict) and "imgs" in batch else batch[0]
            imgs = imgs.to(device, non_blocking=True).float()
            outputs = model(imgs)
            if isinstance(outputs, dict) and "pyramids" in outputs:
                pyramids = outputs["pyramids"]
            elif isinstance(outputs, dict) and "det_maps" in outputs:
                pyramids = outputs["det_maps"]
            elif isinstance(outputs, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in outputs):
                pyramids = outputs
            else:
                pyramids = getattr(outputs, "det_maps", None)
                if pyramids is None:
                    raise RuntimeError("[decode] no pyramids/det_maps found in model outputs")

            head = getattr(model, "head", model)
            try:
                dec_all = _compat_decode_call(head, pyramids, self.score_thresh, self.iou_thresh, self.max_det, self.use_nms)
            except Exception as e:
                print(f"[decode] WARNING head.decode failed: {e}")
                dec_all = dict(boxes=torch.empty(0, 5, device=device),
                               scores=torch.empty(0, device=device),
                               labels=torch.empty(0, dtype=torch.long, device=device))

            if isinstance(dec_all, list):
                dec_list = dec_all
            else:
                dec_list = [dec_all] * imgs.shape[0]

            for i in range(imgs.shape[0]):
                di = dec_list[i] if isinstance(dec_list, list) and len(dec_list) == imgs.shape[0] else dec_all
                boxes = di.get("boxes", torch.empty(0, 5, device=device))
                scores = di.get("scores", torch.empty(0, device=device))
                labels = di.get("labels", torch.empty(0, dtype=torch.long, device=device))
                n_pred_total += int(scores.numel())
                gtb, gtc = _parse_gt_for_image(batch, i, device=device)
                if self.print_debug and batch_idx < 2 and i < 3:
                    if scores.numel():
                        w_med = _to_cpu_numpy(boxes[:, 2]).median()
                        h_med = _to_cpu_numpy(boxes[:, 3]).median()
                        s_med = float(_to_cpu_numpy(scores).mean())
                        print(f"[EVAL IMG {i}] pred={int(scores.numel())} gt={int(gtb.shape[0])}\n"
                              f"  pred w med≈{w_med:.1f}  h med≈{h_med:.1f}  score mean≈{s_med:.3f}")
                    else:
                        print(f"[EVAL IMG {i}] pred=0 gt={int(gtb.shape[0])}\n  pred empty")

                all_preds_np.append(dict(
                    boxes=_to_cpu_numpy(boxes) if boxes.numel() else np.zeros((0, 5), dtype=np.float32),
                    scores=_to_cpu_numpy(scores) if scores.numel() else np.zeros((0,), dtype=np.float32),
                    labels=_to_cpu_numpy(labels).astype(np.int64) if labels.numel() else np.zeros((0,), dtype=np.int64),
                ))
                all_gts_np.append(dict(
                    boxes=_to_cpu_numpy(gtb) if gtb.numel() else np.zeros((0, 5), dtype=np.float32),
                    labels=_to_cpu_numpy(gtc).astype(np.int64) if gtc.numel() else np.zeros((0,), dtype=np.int64),
                ))

        mAP50 = _compute_ap50_per_class(all_preds_np, all_gts_np, int(num_classes), iou_thresh=self.iou_thresh)

        if self.print_debug:
            dur = time.time() - t0
            print(f"[EVAL DEBUG] preds/batch={n_pred_total}  took {dur:.2f}s")
            print(f"[Eval] mAP50={mAP50:.6f} (mode=max)")

        return float(mAP50)
