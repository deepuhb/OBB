import torch, numpy as np
from typing import Any, Dict, List, Union
from ..utils.box_ops import quad_to_cxcywh_angle


def _to_tensor(x, dtype=None):
    if torch.is_tensor(x):
        return x if dtype is None else x.to(dtype=dtype)
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        return t if dtype is None else t.to(dtype=dtype)
    if isinstance(x, (list, tuple)):
        try:
            return torch.tensor(x, dtype=dtype if dtype is not None else torch.float32)
        except Exception:
            return x  # leave as-is if heterogeneous
    return x

def _ensure_targets(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee 'bboxes','labels','kpts' exist as tensors.
       If 'bboxes' missing or not tensor, synthesize from 'quads' else 'boxes+angles'.
    """
    s = sample

    # Coerce common fields to tensors (if present)
    if "labels" in s and not torch.is_tensor(s["labels"]):
        s["labels"] = _to_tensor(s["labels"], dtype=torch.long)
    if "kpts" in s and not torch.is_tensor(s["kpts"]):
        s["kpts"] = _to_tensor(s["kpts"], dtype=torch.float32)
    if "bboxes" in s and not torch.is_tensor(s["bboxes"]):
        s["bboxes"] = _to_tensor(s["bboxes"], dtype=torch.float32)

    # Synthesize bboxes if missing or empty/non-tensor
    need_bboxes = ("bboxes" not in s) or (not torch.is_tensor(s["bboxes"]))
    if not need_bboxes:
        if s["bboxes"].ndim != 2 or s["bboxes"].shape[-1] != 5:
            need_bboxes = True

    if need_bboxes:
        bboxes = None
        # Prefer quads -> OBB
        if "quads" in s:
            q = s["quads"]
            if not torch.is_tensor(q):
                q = _to_tensor(q, dtype=torch.float32)
            if torch.is_tensor(q) and q.numel() > 0:
                q = q.view(-1, 4, 2).float()
                obb = []
                for i in range(q.shape[0]):
                    cx, cy, w, h, th = quad_to_cxcywh_angle(q[i])  # th in radians
                    obb.append([float(cx), float(cy), max(float(w),1.0), max(float(h),1.0), float(th)])
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
                if ang is None or (not torch.is_tensor(ang)) or ang.numel() == 0:
                    th = torch.zeros_like(cx)
                else:
                    th = _to_tensor(ang, dtype=torch.float32).reshape(-1)
                bboxes = torch.stack([cx, cy, w, h, th], dim=1)

        if bboxes is None:
            bboxes = torch.zeros((0, 5), dtype=torch.float32)

        s["bboxes"] = bboxes

    # Final dtype guarantees
    if "labels" not in s or not torch.is_tensor(s["labels"]):
        s["labels"] = torch.zeros((0,), dtype=torch.long)
    if "kpts" not in s or not torch.is_tensor(s["kpts"]):
        s["kpts"] = torch.zeros((0, 2), dtype=torch.float32)

    return s

def _can_stack(lst):
    if not lst or not all(torch.is_tensor(x) for x in lst):
        return False
    s0 = tuple(lst[0].shape)
    return all(tuple(x.shape) == s0 for x in lst)

def collate_obbdet(batch: List[Union[Dict[str, Any], tuple]]):
    # Dict samples (your case)
    if isinstance(batch[0], dict):
        # --- New: pre-normalize each sample so targets are present and tensors ---
        for i in range(len(batch)):
            batch[i] = _ensure_targets(batch[i])

        out: Dict[str, Any] = {}
        keys = set().union(*(b.keys() for b in batch))
        for k in keys:
            vals = [b[k] for b in batch if k in b]
            if k == "image":
                out[k] = torch.stack(vals, dim=0)
                continue
            # AFTER — 'targets' is intentionally omitted
            if k in ("bboxes", "boxes", "quads", "kpts", "labels", "angles",
                     "paths", "meta", "metas", "pos_meta"):
                out[k] = vals
                continue
            # If some wrapper insists on adding 'targets', drop it here:
            if k == "targets":
                continue

            out[k] = torch.stack(vals, dim=0) if _can_stack(vals) else vals
        return out

    # Tuple fallback
    imgs = [b[0] for b in batch]
    tars = [b[1] for b in batch]
    return torch.stack(imgs, dim=0), tars
