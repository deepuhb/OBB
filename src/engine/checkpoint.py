# src/engine/checkpoint.py
from __future__ import annotations

import re
import os
import io
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple, Union

import torch

# PyTorch 2.6 tightened torch.load(); allowlist common numpy types when needed.
try:
    from torch.serialization import add_safe_globals
    import numpy as np
    add_safe_globals([np._core.multiarray.scalar, np.dtype, np.float64, np.float32, np.float16])
except Exception:
    pass


def _unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _extract_state_dict(obj) -> Mapping[str, torch.Tensor]:
    """
    Accepts either a pure state_dict, or a dict with 'model'/'state_dict' keys,
    or a full training bundle. Returns a state_dict.
    """
    if isinstance(obj, Mapping):
        if "model" in obj and isinstance(obj["model"], Mapping):
            return obj["model"]
        if "state_dict" in obj and isinstance(obj["state_dict"], Mapping):
            return obj["state_dict"]
        # some tools save as {"module.xxx": tensor}
        if all(isinstance(k, str) and torch.is_tensor(v) for k, v in obj.items()):
            return obj
    raise ValueError("Unsupported checkpoint format: cannot locate model state_dict")


def _apply_key_maps(sd: Mapping[str, torch.Tensor], maps: Iterable[Tuple[str, str]] | None) -> Dict[str, torch.Tensor]:
    """
    Apply simple string replace maps like [("neck.inner.", "neck.")].
    Returns a NEW dict.
    """
    if not maps:
        return dict(sd)
    out = {}
    for k, v in sd.items():
        new_k = k
        for src, dst in maps:
            if src in new_k:
                new_k = new_k.replace(src, dst)
        out[new_k] = v
    return out


def load_smart_state_dict(
    model: torch.nn.Module,
    ckpt_path: Union[str, os.PathLike],
    map_renames: Iterable[Tuple[str, str]] | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Loads a checkpoint into `model` with key-renaming support and friendly logs.
    Returns a summary dict with counts for matched / skipped / missing.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Try secure load first; fallback to weights_only=False with warning.
    obj = None
    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"[CKPT] torch.load(weights_only=True) failed; falling back to weights_only=False.\n  reason: {e}")
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state = _extract_state_dict(obj)
    state = _apply_key_maps(state, map_renames)

    model_ = _unwrap_ddp(model)
    own = model_.state_dict()

    # Compute intersection and stats
    matched, skipped = 0, 0
    missing = 0
    to_load: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            to_load[k] = v
            matched += 1
        else:
            skipped += 1

    # Find missing (present in model but not provided by ckpt)
    for k in own.keys():
        if k not in state:
            missing += 1

    # Actually load
    msg = model_.load_state_dict(to_load, strict=False)
    if verbose:
        print(f"[CKPT] smart-loaded {matched} tensors; skipped={skipped}; missing(in model)={missing}")
        if len(msg.missing_keys) > 0 or len(msg.unexpected_keys) > 0:
            # Summarize large lists
            def _few(xs):
                return xs[:10] + (["..."] if len(xs) > 10 else [])
            if len(msg.unexpected_keys) > 0:
                print("[CKPT] unexpected (ignored):", _few(msg.unexpected_keys))
            if len(msg.missing_keys) > 0:
                print("[CKPT] missing (left as init):", _few(msg.missing_keys))

    return {"matched": matched, "skipped": skipped, "missing": missing}


def save_checkpoint_bundle(
    path: Union[str, os.PathLike],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    meta: Dict | None = None,
):
    """
    Saves a standard training bundle so eval can reconstruct metrics later.
    Produces a dict with:
      - 'model'     : model.state_dict()  (handles DDP)
      - 'optimizer' : optimizer.state_dict() (if provided)
      - 'scheduler' : scheduler.state_dict() (if provided)
      - 'meta'      : user-provided metadata (dict)
    """
    model_ = _unwrap_ddp(model)
    bundle: Dict[str, object] = {
        "model": model_.state_dict(),
        "meta": meta or {},
    }
    if optimizer is not None:
        bundle["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        bundle["scheduler"] = scheduler.state_dict()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    print(f"[CKPT] saved bundle -> {path}")