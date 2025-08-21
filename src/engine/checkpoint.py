# src/engine/checkpoint.py
import os
import math
import torch
from typing import Dict, Any, Optional

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
