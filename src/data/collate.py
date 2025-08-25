# src/data/collate.py
from __future__ import annotations
from typing import Any, Dict, List, Union
import torch

# Optional util to convert quads->OBB if available
try:
    from src.utils.box_ops import quad_to_cxcywh_angle
except Exception:
    quad_to_cxcywh_angle = None


def _to_tensor(x, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def _coerce_angles_like(cx: torch.Tensor, ang) -> torch.Tensor:
    """Return an angle tensor shaped like `cx`.
    - None or empty -> zeros_like(cx)
    - scalar -> broadcast
    - length mismatch -> zeros_like(cx)
    """
    if ang is None:
        return torch.zeros_like(cx)
    th = _to_tensor(ang, dtype=torch.float32).to(device=cx.device).reshape(-1)
    if th.numel() == 0:
        return torch.zeros_like(cx)
    if th.numel() == 1:
        return th.expand_as(cx)
    if th.numel() != cx.numel():
        return torch.zeros_like(cx)
    return th

def _ensure_targets(s: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single sample so that:
      - image is CHW FloatTensor
      - bboxes (N,5) [cx,cy,w,h,theta_rad] exist (pixels)
      - labels (N,) LongTensor
      - kpts (N,2) FloatTensor (pixels)
    Preserve 'quads','boxes','angles' if present. Drop ambiguous 'targets'.
    """
    # ---- image -> CHW float32 ----
    if "image" not in s:
        raise KeyError("sample missing 'image'")
    img = s["image"]

    if torch.is_tensor(img):
        if img.dim() == 3 and img.shape[0] in (1, 3):  # CHW
            img = img.float()
        elif img.dim() == 3 and img.shape[-1] in (1, 3):  # HWC tensor
            img = img.permute(2, 0, 1).contiguous().float()
        else:
            img = img.contiguous().float()
    else:
        # assume numpy HWC
        import numpy as np
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[2] in (1, 3):
                img = torch.from_numpy(img.transpose(2, 0, 1)).contiguous().float()
            else:
                img = torch.from_numpy(img).contiguous().float()
        else:
            # generic -> tensor
            img = _to_tensor(img).float()

    s["image"] = img

    # ---- labels ----
    s["labels"] = _to_tensor(s.get("labels", []), dtype=torch.long) if "labels" in s else torch.zeros((0,), dtype=torch.long)

    # ---- bboxes (prefer quads->OBB, else boxes+angles) ----
    need_bboxes = ("bboxes" not in s) or (not torch.is_tensor(s["bboxes"]))
    if need_bboxes:
        bboxes = None

        # Prefer quads -> OBB (if util exists)
        if "quads" in s and quad_to_cxcywh_angle is not None:
            q = s["quads"]
            if not torch.is_tensor(q):
                q = _to_tensor(q, dtype=torch.float32)
            if torch.is_tensor(q) and q.numel() > 0:
                q = q.view(-1, 4, 2).float()
                obb = []
                for i in range(q.shape[0]):
                    cx, cy, w, h, th = quad_to_cxcywh_angle(q[i])  # radians
                    obb.append([float(cx), float(cy), max(float(w), 1.0), max(float(h), 1.0), float(th)])
                bboxes = torch.tensor(obb, dtype=torch.float32)

        # Fallback: boxes + angles -> OBB
        if bboxes is None and "boxes" in s:
            bx = s["boxes"]
            if not torch.is_tensor(bx):
                bx = _to_tensor(bx, dtype=torch.float32)
            if torch.is_tensor(bx) and bx.numel() > 0:
                bx = bx.float()
                cx = 0.5 * (bx[:, 0] + bx[:, 2])
                cy = 0.5 * (bx[:, 1] + bx[:, 3])
                w  = (bx[:, 2] - bx[:, 0]).clamp_min_(1.0)
                h  = (bx[:, 3] - bx[:, 1]).clamp_min_(1.0)
                ang = s.get("angles", None)
                th = _coerce_angles_like(cx, ang)
                bboxes = torch.stack([cx, cy, w, h, th], dim=1)

        if bboxes is None:
            bboxes = torch.zeros((0, 5), dtype=torch.float32)

        s["bboxes"] = bboxes

    # ---- kpts ----
    if "kpts" not in s or not torch.is_tensor(s["kpts"]):
        s["kpts"] = torch.zeros((0, 2), dtype=torch.float32)

    # ---- remove ambiguous targets ----
    if "targets" in s:
        s.pop("targets", None)

    return s


def _can_stack(lst):
    if not lst or not all(torch.is_tensor(x) for x in lst):
        return False
    s0 = tuple(lst[0].shape)
    return all(tuple(x.shape) == s0 for x in lst)


def collate_obbdet(batch: List[Union[Dict[str, Any], tuple]]):
    # Dict samples (primary path)
    if isinstance(batch[0], dict):
        batch = [_ensure_targets(b) for b in batch]

        out: Dict[str, Any] = {}
        keys = set().union(*(b.keys() for b in batch))
        for k in keys:
            vals = [b[k] for b in batch if k in b]
            if k == "image":
                out[k] = torch.stack(vals, dim=0)
                continue
            # Keep per-sample lists for ragged keys
            if k in ("bboxes", "boxes", "quads", "kpts", "labels", "angles",
                     "paths", "path", "meta", "metas", "pos_meta"):
                out[k] = vals
                continue
            if k == "targets":
                continue
            out[k] = torch.stack(vals, dim=0) if _can_stack(vals) else vals

        # unify path key
        if "path" in out and "paths" not in out:
            out["paths"] = out["path"]
            out.pop("path", None)
        return out

    # Tuple fallback (img, target)
    imgs = [b[0] for b in batch]
    tars = [b[1] for b in batch]
    return torch.stack(imgs, dim=0), tars
