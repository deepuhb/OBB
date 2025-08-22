# src/engine/checkpoint.py
import os
import math
import numpy as np
import torch

from typing import Dict, Any, Optional, List, Tuple
from torch.serialization import add_safe_globals


# --- metric helpers ---

_MONITOR_ALIASES = {
    "map50": "mAP50",
    "map@0.50": "mAP50",
    "ap50": "mAP50",
    "ap@0.50": "mAP50",
    "map": "mAP",
    "meanap": "mAP",
}

def _norm_key(k: str) -> str:
    k = (k or "").strip()
    if k in _MONITOR_ALIASES:
        return _MONITOR_ALIASES[k]
    k_low = k.lower()
    return _MONITOR_ALIASES.get(k_low, k)  # try lowercase alias, else original

def _is_better(curr: float, best: float, mode: str) -> bool:
    if curr is None or math.isnan(curr):
        return False
    return (curr > best) if mode == "max" else (curr < best)

def _atomic_save(obj: Dict[str, Any], path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _safe_torch_load(path: str):
    """
    Robust torch.load for PyTorch>=2.6:
    - tries weights_only=True by default
    - allowlists needed NumPy globals
    - falls back to weights_only=False if necessary (only if you trust the ckpt)
    """
    try:
        # Newer PyTorch uses numpy._core; older errors mention numpy.core

        try:
            add_safe_globals([np._core.multiarray.scalar, np.dtype])
        except Exception:
            add_safe_globals([np.core.multiarray.scalar, np.dtype])
    except Exception:
        pass

    try:
        return torch.load(path, map_location="cpu")  # PyTorch 2.6 defaults to weights_only=True
    except Exception:
        # Only do this if you trust the checkpoint source.
        return torch.load(path, map_location="cpu", weights_only=False)


def _apply_prefix_map(sd: Dict[str, torch.Tensor], m: Dict[str, str]) -> Dict[str, torch.Tensor]:
    if not m:
        return sd
    out = {}
    for k, v in sd.items():
        nk = k
        for old, new in m.items():
            if nk.startswith(old):
                nk = new + nk[len(old):]
        out[nk] = v
    return out


def _top_prefixes(keys: List[str], k: int = 10) -> List[Tuple[str, int]]:
    from collections import Counter
    return Counter(s.split(".", 1)[0] for s in keys).most_common(k)


def _count_shape_matches(sd_src: Dict[str, torch.Tensor], sd_tgt: Dict[str, torch.Tensor]) -> int:
    cnt = 0
    for k, v in sd_src.items():
        if k in sd_tgt and tuple(sd_tgt[k].shape) == tuple(v.shape):
            cnt += 1
    return cnt


class CheckpointManager:
    """
    Lightweight checkpoint manager with metric-based model selection.
    Compatible with DDP (caller should pass model.module to state_dict when wrapped).
    """

    def __init__(self, save_dir: str, monitor: str = "mAP50", mode: str = "max"):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.monitor = _norm_key(monitor)
        self.mode = mode.lower()
        assert self.mode in ("max", "min")
        self.best_value = -float("inf") if self.mode == "max" else float("inf")

    def update_monitor(self, monitor: str, mode: Optional[str] = None):
        self.monitor = _norm_key(monitor)
        if mode is not None:
            self.mode = mode.lower()
            assert self.mode in ("max", "min")

    def _pack(
        self,
        epoch: int,
        model,
        optimizer,
        scaler,
        cfg,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "metrics": metrics or {},
            "best_metric": self.best_value,
            "monitor": self.monitor,
            "monitor_mode": self.mode,
            "cfg": getattr(cfg, "_raw", None) if cfg is not None else None,
        }

    def save_last(self, epoch: int, model, optimizer, scaler, cfg, metrics) -> str:
        path = os.path.join(self.save_dir, "last.pt")
        _atomic_save(self._pack(epoch, model, optimizer, scaler, cfg, metrics), path)
        return path

    def save_best_if_improved(self, epoch: int, model, optimizer, scaler, cfg, metrics) -> tuple[bool, Optional[str]]:
        # fetch metric (with aliasing)
        mon = self.monitor
        val = metrics.get(mon, None)
        if val is None:
            # try common aliases in the provided metrics
            for k in (mon, mon.lower(), _MONITOR_ALIASES.get(mon, mon)):
                if k in metrics:
                    val = metrics[k]
                    break
        try:
            val = float(val)
        except Exception:
            val = float("nan")

        if _is_better(val, self.best_value, self.mode):
            self.best_value = val
            path = os.path.join(self.save_dir, "best.pt")
            _atomic_save(self._pack(epoch, model, optimizer, scaler, cfg, metrics), path)
            return True, path
        return False, None

    def load(
        self,
        path: str,
        model,
        optimizer=None,
        scaler=None,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> tuple[int, Dict[str, Any]]:
        ckpt = torch.load(path, map_location=map_location)
        # Model
        model.load_state_dict(ckpt["model"], strict=strict)
        # Optimizer / scaler (optional)
        if optimizer is not None and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        # Best metric (several possible keys)
        best = ckpt.get("best_metric", None)
        if best is None:
            m = ckpt.get("metrics", {})
            best = m.get(self.monitor, None)
        try:
            self.best_value = float(best) if best is not None else self.best_value
        except Exception:
            pass
        # Resume epoch (next one to run)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        return start_epoch, ckpt.get("metrics", {})

    def smart_load(
            self,
            path: str,
            model,
            optimizer=None,
            scaler=None,
            img_size: int = 640,
            device: str | torch.device = "cpu",
            strict: bool = False,
            verbose: bool = True,
            extra_maps: Optional[List[Dict[str, str]]] = None,
    ) -> tuple[int, Dict[str, Any]]:
        """
        Warm-build + smart prefix-map loader.
        - Warms the model with a dummy forward so lazy neck/heads exist.
        - Auto-picks the best prefix mapping (e.g., 'neck.inner.'->'neck.', 'head.'->'det_head.', DDP/EMA wrappers).
        - Loads model/optimizer/scaler (non-strict by default).
        Returns: (start_epoch, metrics_dict)
        """
        # Move model to device
        if not isinstance(device, torch.device):
            device = torch.device(device)
        model.to(device)
        model.eval()

        # Warm forward to materialize lazily-built modules (neck/heads)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size, device=device)
            _ = model(dummy)

        # Load ckpt robustly
        ckpt = _safe_torch_load(path)

        # Find state_dict
        sd_raw = None
        if isinstance(ckpt, dict):
            for key in ("model", "state_dict", "ema", "model_ema"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    sd_raw = ckpt[key]
                    break
        if sd_raw is None:
            sd_raw = ckpt if isinstance(ckpt, dict) else {}
        if not isinstance(sd_raw, dict):
            sd_raw = {}

        model_sd = model.state_dict()
        model_keys = list(model_sd.keys())
        ckpt_keys = list(sd_raw.keys())

        # Build candidate prefix maps
        base_maps: List[Dict[str, str]] = [
            {},  # identity
            {"module.": ""},
            {"model.": ""},
            {"ema.": ""},
            {"model_ema.": ""},
        ]

        model_top = {p for p, _ in _top_prefixes(model_keys, k=50)}
        ckpt_top = {p for p, _ in _top_prefixes(ckpt_keys, k=50)}

        # neck inner ↔ plain neck
        if any(k.startswith("neck.inner.") for k in ckpt_keys) and "neck" in model_top:
            base_maps.append({"neck.inner.": "neck."})
        if any(k.startswith("neck.") for k in ckpt_keys) and any(k.startswith("neck.inner.") for k in model_keys):
            base_maps.append({"neck.": "neck.inner."})

        # head ↔ det_head
        if "head" in ckpt_top and "det_head" in model_top:
            base_maps.append({"head.": "det_head."})
        if "det_head" in ckpt_top and "head" in model_top:
            base_maps.append({"det_head.": "head."})

        # user-provided extras
        if extra_maps:
            base_maps.extend(extra_maps)

        # Score candidates, pick best by #shape matches
        best_map, best_score = {}, -1
        tried = set()
        for m in base_maps:
            key = tuple(sorted(m.items()))
            if key in tried:
                continue
            tried.add(key)
            sd_mapped = _apply_prefix_map(sd_raw, m)
            score = _count_shape_matches(sd_mapped, model_sd)
            if score > best_score:
                best_map, best_score = m, score

        # Load model with best map
        sd_mapped = _apply_prefix_map(sd_raw, best_map)
        load_result = model.load_state_dict(sd_mapped, strict=strict)

        missing = getattr(load_result, "missing_keys", None)
        unexpected = getattr(load_result, "unexpected_keys", None)
        if missing is None and unexpected is None and isinstance(load_result, (tuple, list)) and len(load_result) == 2:
            missing, unexpected = load_result
        missing = missing or []
        unexpected = unexpected or []

        # Optimizer / scaler (optional, best-effort)
        if optimizer is not None and isinstance(ckpt, dict) and ckpt.get("optimizer") is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                if verbose:
                    print(f"[CKPT-LOAD] optimizer state load skipped: {e}")
        if scaler is not None and isinstance(ckpt, dict) and ckpt.get("scaler") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                if verbose:
                    print(f"[CKPT-LOAD] scaler state load skipped: {e}")

        # Best metric & resume epoch
        metrics = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}
        best = ckpt.get("best_metric", None) if isinstance(ckpt, dict) else None
        if best is None and metrics:
            best = metrics.get(self.monitor, None)
        try:
            self.best_value = float(best) if best is not None else self.best_value
        except Exception:
            pass
        start_epoch = int(ckpt.get("epoch", -1)) + 1 if isinstance(ckpt, dict) else 0

        if verbose:
            print(f"[CKPT-LOAD] file='{os.path.basename(path)}'  device={device}  img={img_size}")
            print(f"[CKPT-LOAD] best_prefix_map={best_map or '{}'}  matched_by_shape={best_score}")
            if missing:
                print(f"[CKPT-LOAD] missing(in model not in ckpt): {len(missing)} (e.g. {missing[:5]})")
            if unexpected:
                print(f"[CKPT-LOAD] unexpected(in ckpt not in model): {len(unexpected)} (e.g. {unexpected[:5]})")
            print(f"[CKPT-LOAD] model_params={len(model_sd)}  ckpt_params={len(sd_raw)}")

        return start_epoch, metrics
