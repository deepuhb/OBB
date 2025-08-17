# src/data/mosaic_wrapper.py
from __future__ import annotations
import random
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch

# Optional OpenCV
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# Optional Pillow (fallback if no OpenCV)
try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def _rand_hsv(image: np.ndarray, h_gain: float, s_gain: float, v_gain: float) -> np.ndarray:
    if not _HAS_CV2 or (h_gain == 0 and s_gain == 0 and v_gain == 0):
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[..., 0]; s = hsv[..., 1]; v = hsv[..., 2]
    dh = (random.uniform(-h_gain, h_gain) * 180.0)
    ds = random.uniform(1 - s_gain, 1 + s_gain)
    dv = random.uniform(1 - v_gain, 1 + v_gain)
    h = (h + dh) % 180.0
    s = np.clip(s * ds, 0, 255)
    v = np.clip(v * dv, 0, 255)
    hsv[..., 0] = h; hsv[..., 1] = s; hsv[..., 2] = v
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return out


def _ensure_targets_dict(sample: Dict[str, Any], img_w: int, img_h: int) -> Dict[str, Any]:
    """
    Normalize sample to:
      'bboxes' (N,5: cx,cy,w,h,ang), 'labels' (N,), optional 'kpts' (N,1,2)
    Accepts either YOLO 'targets' or separate fields. Denormalizes if in [0,1].
    """
    if "bboxes" in sample and "labels" in sample:
        boxes = np.asarray(sample["bboxes"], dtype=np.float32)
        labels = np.asarray(sample["labels"], dtype=np.int64)
        kpts = sample.get("kpts", None)
        if kpts is not None:
            kpts = np.asarray(kpts, dtype=np.float32)
    elif "targets" in sample and isinstance(sample["targets"], (torch.Tensor, np.ndarray)):
        t = sample["targets"]
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        if t.size == 0:
            boxes = np.zeros((0, 5), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            kpts = None
        else:
            cls = t[:, 1].astype(np.int64)
            cx, cy, w, h, ang = t[:, 2], t[:, 3], t[:, 4], t[:, 5], t[:, 6]
            boxes = np.stack([cx, cy, w, h, ang], axis=1).astype(np.float32)
            labels = cls
            if t.shape[1] >= 9:
                kp = t[:, 7:9].astype(np.float32).reshape(-1, 1, 2)
                kpts = kp
            else:
                kpts = None
    else:
        boxes = np.zeros((0, 5), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int64)
        kpts = None

    # Denormalize if necessary
    if boxes.size and np.all((boxes[:, :4] >= 0) & (boxes[:, :4] <= 1.0)):
        boxes[:, 0] *= img_w; boxes[:, 1] *= img_h
        boxes[:, 2] *= img_w; boxes[:, 3] *= img_h
    if kpts is not None and kpts.size and np.all((kpts >= 0) & (kpts <= 1.0)):
        kpts[..., 0] *= img_w; kpts[..., 1] *= img_h

    return {"bboxes": boxes, "labels": labels, "kpts": kpts}


def _to_chw_tensor(image_rgb_u8: np.ndarray) -> torch.Tensor:
    # HWC uint8 -> CHW float32 in [0,1]
    return torch.from_numpy(image_rgb_u8).permute(2, 0, 1).contiguous().float().div_(255.0)


def _maybe_to_chw_like_base(base_sample: Dict[str, Any], image_rgb_u8: np.ndarray):
    """Match base image type: if base was tensor, return CHW tensor; else CHW numpy."""
    base_img = base_sample["image"]
    if isinstance(base_img, torch.Tensor):
        return _to_chw_tensor(image_rgb_u8)
    else:
        # return numpy CHW
        return np.ascontiguousarray(np.transpose(image_rgb_u8, (2, 0, 1)))


def _resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    w, h = size  # (W, H)
    if _HAS_CV2:
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    if _HAS_PIL:
        return np.array(Image.fromarray(image).resize((w, h), Image.BILINEAR))
    # ultra-basic nearest fallback
    sy = np.linspace(0, image.shape[0] - 1, h).astype(np.int32)
    sx = np.linspace(0, image.shape[1] - 1, w).astype(np.int32)
    return image[sy][:, sx]


def _pack_back_like_base(
    base_sample: Dict[str, Any],
    image_rgb_u8: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    kpts: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Return a dict matching the base sample’s supervision format, with CHW image."""
    out = dict(base_sample)
    out["image"] = _maybe_to_chw_like_base(base_sample, image_rgb_u8)

    if "targets" in base_sample:
        N = boxes.shape[0]
        if N == 0:
            targets = np.zeros((0, 7), dtype=np.float32)
        else:
            img_idx = np.zeros((N, 1), dtype=np.float32)   # collate_fn can overwrite indices per-batch
            cls = labels.reshape(-1, 1).astype(np.float32)
            t = np.concatenate([img_idx, cls, boxes.astype(np.float32)], axis=1)  # (N,7)
            if kpts is not None and kpts.size:
                t = np.concatenate([t, kpts.reshape(N, 2)], axis=1)
            targets = t
        out["targets"] = torch.from_numpy(targets).float()
        out.pop("bboxes", None); out.pop("labels", None); out.pop("kpts", None)
    else:
        out["bboxes"] = boxes.astype(np.float32)
        out["labels"] = labels.astype(np.int64)
        if kpts is not None:
            out["kpts"] = kpts.astype(np.float32)
        else:
            out.pop("kpts", None)
        out.pop("targets", None)

    return out


class AugmentingDataset:
    """
    Mosaic + flips + light HSV wrapper. Preserves base dataset's label format.
    Always returns images in **CHW** (torch tensor if base was tensor).
    """

    def __init__(
        self,
        base,
        *,
        mosaic: bool = True,
        mosaic_prob: float = 0.5,
        fliplr: float = 0.5,
        flipud: float = 0.0,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
    ):
        self.base = base
        self.mosaic_enabled = bool(mosaic)
        self.mosaic_prob = float(mosaic_prob)
        self.fliplr = float(fliplr)
        self.flipud = float(flipud)
        self.hsv_h = float(hsv_h)
        self.hsv_s = float(hsv_s)
        self.hsv_v = float(hsv_v)   # <-- fixed: no keyword arg
        # runtime toggle (trainer turns this off in final 10%)
        self.mosaic_active = self.mosaic_enabled
        # expose names/nc if base has them
        self.names = getattr(base, "names", None)
        self.nc = getattr(base, "nc", None)
        self.collate_fn = getattr(base, "collate_fn", None)
        self.img_size = getattr(base, "img_size", None)

    def __len__(self):
        return len(self.base)

    def set_mosaic_active(self, active: bool):
        self.mosaic_active = bool(active)

    def _load_base(self, idx: int) -> Dict[str, Any]:
        s = self.base[idx]
        img = s["image"]
        # Convert to HWC uint8 RGB for augmentation
        if isinstance(img, torch.Tensor):
            # assume CHW float [0,1] or [0,255]
            arr = img.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW
                if arr.max() <= 1.0:
                    arr = (arr * 255.0)
                arr = arr.transpose(1, 2, 0)
            im = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            im = img
            if im.dtype != np.uint8:
                im = np.clip(im, 0, 255).astype(np.uint8)
        h, w = im.shape[:2]
        norm = _ensure_targets_dict(s, w, h)
        return {"image": im, **norm, "raw": s}

    def _flip_and_hsv(
        self, image: np.ndarray, boxes: np.ndarray, kpts: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        h, w = image.shape[:2]
        # HSV jitter
        image = _rand_hsv(image, self.hsv_h, self.hsv_s, self.hsv_v)

        # Horizontal flip
        if random.random() < self.fliplr:
            image = np.ascontiguousarray(image[:, ::-1, :])
            if boxes.size:
                boxes[:, 0] = w - boxes[:, 0]     # flip cx around center
                boxes[:, 4] = -boxes[:, 4]        # mirror angle (approx.)
            if kpts is not None and kpts.size:
                kpts[..., 0] = w - kpts[..., 0]

        # Vertical flip
        if random.random() < self.flipud:
            image = np.ascontiguousarray(image[::-1, :, :])
            if boxes.size:
                boxes[:, 1] = h - boxes[:, 1]     # flip cy
                boxes[:, 4] = -boxes[:, 4]        # mirror angle again
            if kpts is not None and kpts.size:
                kpts[..., 1] = h - kpts[..., 1]

        return image, boxes, kpts

    def __getitem__(self, index: int) -> Dict[str, Any]:
        do_mosaic = self.mosaic_active and self.mosaic_enabled and (random.random() < self.mosaic_prob)
        base0 = self._load_base(index)
        img0, boxes0, labels0, kpts0 = base0["image"], base0["bboxes"], base0["labels"], base0["kpts"]

        if not do_mosaic:
            img, boxes, kpts = self._flip_and_hsv(img0.copy(), boxes0.copy(), None if kpts0 is None else kpts0.copy())
            return _pack_back_like_base(base0["raw"], img, boxes, labels0, kpts)

        # ---- 4-image mosaic ----
        side = int(self.img_size) if self.img_size else max(img0.shape[0], img0.shape[1])
        H = W = side
        half = side // 2

        idxs = [index] + [random.randint(0, len(self.base) - 1) for _ in range(3)]
        imgs: List[np.ndarray] = []
        boxes_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        kpts_list: List[Optional[np.ndarray]] = []

        for idx in idxs:
            b = self._load_base(idx)
            im = b["image"]
            # resize each tile to (half, half)
            imr = _resize(im, (half, half))
            ih, iw = im.shape[:2]
            scale_x = half / max(1, iw)
            scale_y = half / max(1, ih)

            bx = b["bboxes"].copy()
            if bx.size:
                bx[:, 0] *= scale_x
                bx[:, 1] *= scale_y
                bx[:, 2] *= scale_x
                bx[:, 3] *= scale_y
            kp = b["kpts"]
            if kp is not None and kp.size:
                kp = kp.copy()
                kp[..., 0] *= scale_x
                kp[..., 1] *= scale_y

            imgs.append(imr)
            boxes_list.append(bx)
            labels_list.append(b["labels"].copy())
            kpts_list.append(kp)

        canvas = np.full((H, W, 3), 114, dtype=np.uint8)
        offsets = [(0, 0), (0, half), (half, 0), (half, half)]
        all_boxes = []
        all_labels = []
        all_kpts = []

        for tile_id, (imr, bx, lbls, kp) in enumerate(zip(imgs, boxes_list, labels_list, kpts_list)):
            oy, ox = offsets[tile_id]
            canvas[oy:oy+half, ox:ox+half] = imr
            if bx.size:
                bx2 = bx.copy()
                bx2[:, 0] += ox
                bx2[:, 1] += oy
                all_boxes.append(bx2)
                all_labels.append(lbls)
                if kp is not None and kp.size:
                    kp2 = kp.copy()
                    kp2[..., 0] += ox
                    kp2[..., 1] += oy
                    all_kpts.append(kp2)

        if len(all_boxes) == 0:
            boxes = np.zeros((0, 5), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            kpts = None
        else:
            boxes = np.concatenate(all_boxes, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            kpts = np.concatenate(all_kpts, axis=0) if len(all_kpts) else None

        img, boxes, kpts = self._flip_and_hsv(canvas, boxes, kpts)
        return _pack_back_like_base(base0["raw"], img, boxes, labels, kpts)
