# src/models/postprocess/decode.py
import torch
import torchvision

@torch.no_grad()
def decode(det_maps, kpt_maps, strides,
           score_thresh, nms_iou, max_det, topk, apply_nms):
    """
    Returns list of dicts per image:
      {"boxes": (N,4) xyxy, "scores": (N,), "labels": (N,), "kpts": (N,2)}
    """
    device = det_maps[0].device
    B = det_maps[0].shape[0]
    outs = []

    for b in range(B):
        boxes_all, scores_all, labels_all, kpts_all = [], [], [], []
        for dm, km, s in zip(det_maps, kpt_maps, strides):
            logits = dm[b]  # (C,H,W)
            H, W = logits.shape[1:]

            tx = logits[0].sigmoid()
            ty = logits[1].sigmoid()
            tw = logits[2].exp()
            th = logits[3].exp()
            obj = logits[6].sigmoid()

            ys, xs = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij"
            )
            cx = (tx + xs) * s
            cy = (ty + ys) * s
            w  = tw.clamp_min(1e-3) * s
            h  = th.clamp_min(1e-3) * s

            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h

            keep = obj > score_thresh
            if keep.sum() == 0:
                continue

            xx1 = x1[keep]; yy1 = y1[keep]; xx2 = x2[keep]; yy2 = y2[keep]
            sc  = obj[keep]

            uv = km[b].permute(1, 2, 0).sigmoid()
            u = uv[..., 0][keep]; v = uv[..., 1][keep]
            kx = (u - 0.5) * (xx2 - xx1) + 0.5 * (xx1 + xx2)
            ky = (v - 0.5) * (yy2 - yy1) + 0.5 * (yy1 + yy2)

            boxes_all.append(torch.stack([xx1, yy1, xx2, yy2], dim=-1))
            scores_all.append(sc)
            labels_all.append(torch.zeros_like(sc, dtype=torch.long))
            kpts_all.append(torch.stack([kx, ky], dim=-1))

        if len(scores_all) == 0:
            outs.append({"boxes": torch.zeros((0,4), device=device),
                         "scores": torch.zeros((0,), device=device),
                         "labels": torch.zeros((0,), dtype=torch.long, device=device),
                         "kpts": torch.zeros((0,2), device=device)})
            continue

        boxes  = torch.cat(boxes_all, 0)
        scores = torch.cat(scores_all, 0)
        labels = torch.cat(labels_all, 0)
        kpts   = torch.cat(kpts_all, 0)

        # pre-cap by score
        if topk is not None and scores.numel() > topk:
            idx = torch.topk(scores, k=topk, largest=True, sorted=True).indices
            boxes, scores, labels, kpts = boxes[idx], scores[idx], labels[idx], kpts[idx]

        if apply_nms:
            keep = torchvision.ops.nms(boxes, scores, nms_iou)
            if keep.numel() > max_det:
                keep = keep[:max_det]
            boxes, scores, labels, kpts = boxes[keep], scores[keep], labels[keep], kpts[keep]
        else:
            if scores.numel() > max_det:
                idx = torch.topk(scores, k=max_det, largest=True, sorted=True).indices
                boxes, scores, labels, kpts = boxes[idx], scores[idx], labels[idx], kpts[idx]

        outs.append({"boxes": boxes, "scores": scores, "labels": labels, "kpts": kpts})
    return outs
