# src/data/mosaic_wrapper.py
from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from .transforms import transform_sample  # for final norm/CHW conversion


def _aabb_from_quads(quads: np.ndarray) -> np.ndarray:
    """quads: (N,4,2) -> boxes (N,4) [x1,y1,x2,y2]."""
    if quads.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x1 = quads[:, :, 0].min(axis=1)
    y1 = quads[:, :, 1].min(axis=1)
    x2 = quads[:, :, 0].max(axis=1)
    y2 = quads[:, :, 1].max(axis=1)
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


def _augment_hsv(img: np.ndarray, hgain=0.015, sgain=0.7, vgain=0.4):
    if hgain == 0 and sgain == 0 and vgain == 0:
        return img
    r = np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain]) + 1.0
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    dtype = img.dtype  # uint8
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    img = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB).astype(dtype)
    return img


class AugmentingDataset(torch.utils.data.Dataset):
    """
    YOLO11-style mosaic(+mixup) wrapper around a base dataset (which must expose .items list of (img_path, lab_path)
    and methods _read_image(path) and _parse_label_file(path, w, h)).
    Produces final (3, S, S) tensor images, with quads/kpts in pixel coords.
    """

    def __init__(
        self,
        base,
        mosaic: bool = True,
        mosaic_prob: float = 0.5,
        fliplr: float = 0.5,
        flipud: float = 0.0,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        mixup_prob: float = 0.1,
        mixup_alpha: float = 0.2,
    ):
        self.base = base
        self.mosaic = mosaic
        self.mosaic_prob = float(mosaic_prob)
        self.fliplr = float(fliplr)
        self.flipud = float(flipud)
        self.hsv_h = float(hsv_h)
        self.hsv_s = float(hsv_s)
        self.hsv_v = float(hsv_v)
        self.mixup_prob = float(mixup_prob)
        self.mixup_alpha = float(mixup_alpha)

        # infer target size from base
        self.size = int(getattr(base, "img_size", 640))

    def __len__(self):
        return len(self.base)

    def _load_raw(self, index: int):
        """Read one raw image + annotations in pixel coords from the base dataset."""
        img_path, lab_path = self.base.items[index]
        img = self.base._read_image(img_path)
        h, w = img.shape[:2]
        quads_list, kpts_list, labels_list = self.base._parse_label_file(lab_path, w, h)
        quads = np.stack(quads_list, axis=0).astype(np.float32) if len(quads_list) else np.zeros((0, 4, 2), np.float32)
        kpts = np.asarray(kpts_list, dtype=np.float32) if len(kpts_list) else np.zeros((0, 2), np.float32)
        labels = np.asarray(labels_list, dtype=np.int64) if len(labels_list) else np.zeros((0,), np.int64)
        return img, quads, kpts, labels

    def _scale_img_ann(self, img: np.ndarray, quads: np.ndarray, kpts: np.ndarray, scale: float):
        if scale == 1.0:
            return img, quads, kpts
        nh, nw = int(round(img.shape[0] * scale)), int(round(img.shape[1] * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if quads.size:
            quads = quads * float(scale)
        if kpts.size:
            kpts = kpts * float(scale)
        return img, quads, kpts

    def _place(self, mosaic_img: np.ndarray, img: np.ndarray, xc: int, yc: int, k: int):
        """Compute placement coords (mosaic and image crop) given center (xc,yc) and tile index k."""
        s = mosaic_img.shape[0] // 2  # since canvas is (2s,2s)
        h, w = img.shape[:2]
        if k == 0:  # top-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif k == 1:  # top-right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif k == 2:  # bottom-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, 2 * s)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
        else:  # k == 3, bottom-right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(yc + h, 2 * s)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)
        return (x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b)

    def _mosaic(self, index: int):
        s = int(self.size)
        # final canvas 2s x 2s
        mosaic_img = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)
        # mosaic center
        yc = int(random.uniform(s // 2, 3 * s // 2))
        xc = int(random.uniform(s // 2, 3 * s // 2))

        indices = [index] + [random.randint(0, len(self.base) - 1) for _ in range(3)]
        quads_all, kpts_all, labels_all = [], [], []

        for k, idx in enumerate(indices):
            img, quads, kpts, labels = self._load_raw(idx)

            # scale factor to make long side ~ s
            scale = s / max(img.shape[:2])
            # optional random scale jitter like YOLO (0.5~1.5)
            scale *= random.uniform(0.5, 1.5)
            img, quads, kpts = self._scale_img_ann(img, quads, kpts, scale)

            (x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b) = self._place(mosaic_img, img, xc, yc, k)
            # paste
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            padx, pady = x1a - x1b, y1a - y1b
            if quads.size:
                q = quads.copy()
                q[:, :, 0] += padx
                q[:, :, 1] += pady
                quads_all.append(q)
            if kpts.size:
                p = kpts.copy()
                p[:, 0] += padx
                p[:, 1] += pady
                kpts_all.append(p)
            if labels.size:
                labels_all.append(labels.copy())

        quads_all = np.concatenate(quads_all, axis=0) if len(quads_all) else np.zeros((0, 4, 2), np.float32)
        kpts_all = np.concatenate(kpts_all, axis=0) if len(kpts_all) else np.zeros((0, 2), np.float32)
        labels_all = np.concatenate(labels_all, axis=0) if len(labels_all) else np.zeros((0,), np.int64)

        # crop final image to s x s
        x0 = max(0, xc - s // 2); y0 = max(0, yc - s // 2)
        x1 = x0 + s; y1 = y0 + s
        mosaic_img = mosaic_img[y0:y1, x0:x1]
        # adjust coords
        if quads_all.size:
            quads_all[:, :, 0] -= x0
            quads_all[:, :, 1] -= y0
        if kpts_all.size:
            kpts_all[:, 0] -= x0
            kpts_all[:, 1] -= y0
        # clip to image bounds
        if quads_all.size:
            quads_all[:, :, 0] = np.clip(quads_all[:, :, 0], 0, s - 1)
            quads_all[:, :, 1] = np.clip(quads_all[:, :, 1], 0, s - 1)
        if kpts_all.size:
            kpts_all[:, 0] = np.clip(kpts_all[:, 0], 0, s - 1)
            kpts_all[:, 1] = np.clip(kpts_all[:, 1], 0, s - 1)

        return mosaic_img, quads_all, kpts_all, labels_all

    def _maybe_flip(self, img: np.ndarray, quads: np.ndarray, kpts: np.ndarray):
        h, w = img.shape[:2]
        # horizontal flip
        if self.fliplr > 0 and random.random() < self.fliplr:
            img = cv2.flip(img, 1)
            if quads.size:
                quads[:, :, 0] = w - quads[:, :, 0]
            if kpts.size:
                kpts[:, 0] = w - kpts[:, 0]
        # vertical flip
        if self.flipud > 0 and random.random() < self.flipud:
            img = cv2.flip(img, 0)
            if quads.size:
                quads[:, :, 1] = h - quads[:, :, 1]
            if kpts.size:
                kpts[:, 1] = h - kpts[:, 1]
        return img, quads, kpts

    def _maybe_mixup(self, img: np.ndarray, quads: np.ndarray, kpts: np.ndarray, labels: np.ndarray):
        if self.mixup_prob <= 0 or random.random() >= self.mixup_prob:
            return img, quads, kpts, labels
        # sample another mosaic image of same size
        j = random.randint(0, len(self.base) - 1)
        img2, quads2, kpts2, labels2 = self._mosaic(j) if self.mosaic else self._load_raw(j)
        # Ensure img2 shape == img
        if img2.shape[:2] != img.shape[:2]:
            img2 = cv2.resize(img2, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        a = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        img = (img.astype(np.float32) * a + img2.astype(np.float32) * (1 - a)).astype(np.uint8)
        if quads2.size:
            quads = np.concatenate([quads, quads2], axis=0) if quads.size else quads2
        if kpts2.size:
            kpts = np.concatenate([kpts, kpts2], axis=0) if kpts.size else kpts2
        if labels2.size:
            labels = np.concatenate([labels, labels2], axis=0) if labels.size else labels2
        return img, quads, kpts, labels

    def __getitem__(self, index: int) -> Dict:
        use_mosaic = self.mosaic and (random.random() < self.mosaic_prob)
        if use_mosaic:
            img, quads, kpts, labels = self._mosaic(index)
            # flip + hsv on the composed image
            img, quads, kpts = self._maybe_flip(img, quads, kpts)
            img = _augment_hsv(img, self.hsv_h, self.hsv_s, self.hsv_v)
            # optional mixup
            img, quads, kpts, labels = self._maybe_mixup(img, quads, kpts, labels)
            # build sample and run final normalize
            sample = {
                "image": img,
                "quads": quads,
                "boxes": _aabb_from_quads(quads),
                "kpts": kpts,
                "labels": labels,
                "angles": np.zeros((labels.shape[0],), dtype=np.float32),
                "path": None,
                "meta": {"orig_size": img.shape[:2], "mosaic": True},
            }
            sample = transform_sample(sample, self.size)  # scale=1 if size already matches
            return sample

        # Fallback: base dataset sample (already resized/normalized)
        return self.base[index]
