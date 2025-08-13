# src/engine/checkpoint.py
import os
import math
import torch
from typing import Dict, Any

def _is_better(curr: float, best: float, mode: str) -> bool:
    if math.isnan(curr):
        return False
    return (curr > best) if mode == "max" else (curr < best)

class CheckpointManager:
    def __init__(self, save_dir: str, monitor: str = "map50", mode: str = "max"):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode.lower()
        assert self.mode in ("max", "min")
        self.best_value = -float("inf") if self.mode == "max" else float("inf")

    def state_dict(self, epoch: int, model, optimizer, scaler, cfg, metrics: Dict[str, Any]):
        # for DDP-wrapped models, caller should pass the .module
        return {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "metrics": metrics,
            "best_metric": self.best_value,
            "monitor": self.monitor,
            "monitor_mode": self.mode,
            "cfg": getattr(cfg, "_raw", None),  # raw YAML dict if available
        }

    def save_last(self, epoch: int, model, optimizer, scaler, cfg, metrics):
        path = os.path.join(self.save_dir, "last.pt")
        torch.save(self.state_dict(epoch, model, optimizer, scaler, cfg, metrics), path)
        return path

    def save_best_if_improved(self, epoch: int, model, optimizer, scaler, cfg, metrics):
        val = float(metrics.get(self.monitor, float("nan")))
        if _is_better(val, self.best_value, self.mode):
            self.best_value = val
            path = os.path.join(self.save_dir, "best.pt")
            torch.save(self.state_dict(epoch, model, optimizer, scaler, cfg, metrics), path)
            return True, path
        return False, None

    def load(self, path: str, model, optimizer=None, scaler=None, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        self.best_value = ckpt.get("best_metric", self.best_value)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        metrics = ckpt.get("metrics", {})
        return start_epoch, metrics
