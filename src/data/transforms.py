
# src/data/transforms.py
import numpy as np
import torch
import cv2

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_longest_edge(img, size):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_h = size - nh
    pad_w = size - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, scale, (left, top)


def transform_sample(sample, img_size):
    """
    Resizes/pads to square 'img_size', normalizes to ImageNet stats,
    preserves 'meta' and 'path', and returns tensors where possible.
    Expects 'boxes','quads','kpts','angles','labels' in pixel coords.
    """
    img = sample['image']
    boxes = sample.get('boxes', None)
    quads = sample.get('quads', None)
    kpts  = sample.get('kpts',  None)
    angles= sample.get('angles', None)
    labels= sample.get('labels', None)

    # keep passthrough
    meta = sample.get('meta', {})
    path = sample.get('path', sample.get('paths', None))

    im, scale, (dx, dy) = resize_longest_edge(img, img_size)

    # Defaults
    import numpy as _np
    boxes = _np.zeros((0,4), _np.float32) if boxes is None else boxes
    quads = _np.zeros((0,4,2), _np.float32) if quads is None else quads
    kpts  = _np.zeros((0,2), _np.float32) if kpts is None else kpts
    angles= _np.zeros((0,), _np.float32) if angles is None else angles
    labels= _np.zeros((0,), _np.int64) if labels is None else labels

    # transform annotations
    if len(boxes) > 0:
        boxes = _np.asarray(boxes, dtype=_np.float32) * scale
        boxes[:, [0, 2]] += dx
        boxes[:, [1, 3]] += dy
    if len(quads) > 0:
        quads = _np.asarray(quads, dtype=_np.float32) * scale
        quads[:, :, 0] += dx
        quads[:, :, 1] += dy
    if len(kpts) > 0:
        kpts = _np.asarray(kpts, dtype=_np.float32) * scale
        kpts[:, 0] += dx
        kpts[:, 1] += dy

    # to float CHW (normalized)
    im = im.astype(_np.float32) / 255.0
    im = (im - _IMAGENET_MEAN) / _IMAGENET_STD
    im = torch.from_numpy(im.transpose(2, 0, 1)).float().contiguous()

    # pack back, preserving meta/path
    out = {
        'image': im,
        'boxes': torch.from_numpy(boxes).float() if not torch.is_tensor(boxes) else boxes,
        'quads': torch.from_numpy(quads).float() if not torch.is_tensor(quads) else quads,
        'kpts':  torch.from_numpy(kpts).float()  if not torch.is_tensor(kpts)  else kpts,
        'angles': torch.from_numpy(angles).float() if not torch.is_tensor(angles) else angles,
        'labels': torch.from_numpy(labels).long() if not torch.is_tensor(labels) else labels,
        'meta': meta,
    }
    if path is not None:
        out['path'] = path
    return out
