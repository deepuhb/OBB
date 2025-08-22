# Copyright (c) 2025
# Robust training loop with AMP + (optional) DDP and flexible criterion API.
from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from numbers import Number

import torch
import torch.nn as nn
import torch.distributed as dist


Tensor = torch.Tensor


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _ddp_mean_scalar(x: float) -> float:
    """Average a scalar across ranks for logging."""
    if not _is_dist():
        return x
    t = torch.tensor([x], dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t.item() / float(_world_size()))


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model if wrapped by DDP/DataParallel."""
    return getattr(model, "module", model)


def _stack_if_list(x: Union[Tensor, Sequence[Tensor]]) -> Tensor:
    if isinstance(x, Tensor):
        return x
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Empty image list in batch.")
        if all(isinstance(t, Tensor) for t in x):
            # Accept either [B,C,H,W] already or list of [C,H,W].
            if x[0].ndim == 3:  # [C,H,W] -> stack to [B,C,H,W]
                return torch.stack(x, dim=0)
            elif x[0].ndim == 4:
                return torch.stack(x, dim=0)  # rare case of per-sample batched tensors
    raise TypeError(f"Unsupported image container type: {type(x)}")

def _as_float(x):
    # Accepts float/int, numpy scalar, or torch.Tensor
    if isinstance(x, Number):
        return float(x)
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        # If someone accidentally returns a vector/tensor metric, reduce safely
        return x.detach().mean().item()
    try:
        # numpy scalar or anything float-able
        return float(x)
    except Exception:
        # last resort: stringify warning
        print(f"[WARN] metric '{x}' not scalar-like; forcing mean/0.0", flush=True)
        return 0.0


def _extract_images_from_batch(batch: Dict[str, Any]) -> Tensor:
    """Find images in common keys without boolean-chaining on tensors."""
    for k in ("image", "images", "img", "imgs"):
        if k in batch and batch[k] is not None:
            imgs = batch[k]
            return _stack_if_list(imgs)
    # Some datasets put it under 'inputs'
    if "inputs" in batch and batch["inputs"] is not None:
        return _stack_if_list(batch["inputs"])
    raise KeyError("Could not find image tensor in batch. Tried: image/images/img/imgs/inputs")


class Trainer:
    """
    Minimal, robust Trainer used by scripts/train.py.

    - AMP via torch.amp.autocast + GradScaler
    - Optional DDP aware logging/sync
    - Flexible criterion output handling:
        * criterion(...) -> Tensor total_loss
        * criterion(...) -> dict with 'total' and/or individual loss parts
        * criterion(...) -> (Tensor total_loss, dict parts)
    - No checkpoint saving here (train.py owns it).

    Expected model forward:
        outputs = model(images) -> (det_maps, feats, *optional)
    Expected criterion forward:
        criterion(det_maps, feats, batch, model=core_model, epoch=epoch) -> ...
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
        cfg: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg or {}
        self.logger = logger

        # DDP info
        self.rank = _rank()
        self.world_size = _world_size()
        self.is_main = (self.rank == 0)

        # AMP
        use_amp = bool(self.cfg.get("amp", False))
        if scaler is not None:
            self.scaler = scaler
        else:
            if use_amp and device.type == "cuda":
                # torch>=2.4 uses torch.amp.GradScaler(device)
                try:
                    self.scaler = torch.amp.GradScaler("cuda")
                except TypeError:
                    # Older signature
                    self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            else:
                # disabled scaler acts like identity
                try:
                    self.scaler = torch.amp.GradScaler("cuda", enabled=False)
                except TypeError:
                    self.scaler = torch.cuda.amp.GradScaler(enabled=False)

    # -------------------------
    #  public API
    # -------------------------
    def fit(
        self,
        train_loader: Iterable[Dict[str, Any]],
        val_loader: Optional[Iterable[Dict[str, Any]]] = None,
        evaluator: Optional[Any] = None,
        train_sampler: Optional[Any] = None,
    ) -> Optional[float]:
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler

        # Epoch settings
        epochs = int(self.cfg.get("epochs", 1))
        log_interval = int(self.cfg.get("log_interval", 50))
        use_amp = bool(self.cfg.get("amp", False)) and self.device.type == "cuda"

        best_metric: Optional[float] = None
        model.train()

        for epoch in range(1, epochs + 1):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            # -----------------
            #   Train epoch
            # -----------------
            t0 = time.time()
            running_total = 0.0
            parts_accum: Dict[str, float] = {}
            steps = 0

            for step, batch in enumerate(train_loader, start=1):
                imgs = _extract_images_from_batch(batch)
                if imgs.device.type != self.device.type:
                    imgs = imgs.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # Forward
                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = model(imgs)
                        loss, parts = self._call_criterion(criterion, outputs, batch, epoch)
                else:
                    outputs = model(imgs)
                    loss, parts = self._call_criterion(criterion, outputs, batch, epoch)

                # Backward
                if self.scaler and self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                # Logging accumulators (CPU floats)
                loss_item = float(loss.detach().item())
                running_total += loss_item
                parts_accum = _merge_add(parts_accum, parts)
                steps += 1

                if log_interval > 0 and (step % log_interval == 0):
                    avg_loss = running_total / max(1, steps)
                    if self.is_main:
                        print(f"[rank{self.rank}] epoch {epoch} step {step}: loss={avg_loss:.4f}", flush=True)

            # End epoch logging
            epoch_loss = running_total / max(1, steps)
            # Average across DDP ranks for cleaner logs
            epoch_loss_avg = _ddp_mean_scalar(epoch_loss)

            if self.is_main:
                print(f"[{_ts()}] Epoch {epoch}: loss={epoch_loss_avg:.4f}", flush=True)

            # Step scheduler (per-epoch)
            if scheduler is not None:
                try:
                    scheduler.step()
                except Exception:
                    # Some schedulers require metric: pass epoch_loss_avg
                    try:
                        scheduler.step(epoch_loss_avg)
                    except Exception:
                        pass

            # -----------------
            #   Evaluate (main rank only)
            # -----------------
            if evaluator is not None and val_loader is not None:
                if _is_dist():
                    dist.barrier()
                metrics = None
                if self.is_main:
                    model.eval()
                    with torch.inference_mode():
                        # Evaluator can wrap its own autocast if needed.
                        metrics = evaluator.evaluate(model, val_loader, device=self.device)
                    model.train()

                    # Basic display of a chosen metric (scalar-safe)
                    if isinstance(metrics, dict):
                        select_name = str(self.cfg.get("eval_select", "mAP50"))
                        sel_raw = metrics.get(select_name, None)
                        if sel_raw is not None:
                            sel = _as_float(sel_raw)
                            print(f"[{_ts()}] Eval: {select_name}={sel:.6f}  (mode={self.cfg.get('eval_mode', 'max')})",
                                  flush=True)
                            # Track best on main rank
                            if best_metric is None:
                                best_metric = sel
                            else:
                                mode = str(self.cfg.get("eval_mode", "max"))
                                better = (sel > best_metric) if mode == "max" else (sel < best_metric)
                                if better:
                                    best_metric = sel
                        else:
                            print(f"[{_ts()}] Eval: key '{select_name}' missing; metrics={metrics}", flush=True)
                    else:
                        print(f"[{_ts()}] Eval returned non-dict: {type(metrics)}", flush=True)

                # Sync after eval
                if _is_dist():
                    dist.barrier()

            if self.is_main:
                # brief epoch timing
                print(f"[{_ts()}] Epoch {epoch} finished in {(time.time()-t0):.2f}s", flush=True)

        return best_metric

    # -------------------------
    #  helpers
    # -------------------------
    def _call_criterion(
        self,
        criterion: nn.Module,
        outputs: Any,
        batch: Dict[str, Any],
        epoch: int,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Call criterion flexibly and return (loss_tensor, parts_dict_of_floats)."""
        # Unwrap the model so criterion can access custom methods (e.g., kpt_from_obbs)
        core_model = _unwrap_model(self.model)

        # Unpack common output shapes
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
            det_maps, feats = outputs[0], outputs[1]
        else:
            raise TypeError(
                "Model forward must return at least (det_maps, feats). "
                f"Got type={type(outputs)} len={len(outputs) if isinstance(outputs,(list,tuple)) else 'n/a'}"
            )

        result = criterion(det_maps, feats, batch, model=core_model, epoch=epoch)

        # Normalize result to (Tensor loss, Dict[str,float] parts)
        if isinstance(result, tuple) and len(result) == 2:
            loss, parts = result
            loss = _ensure_tensor(loss, device=self.device)
            parts = _parts_to_floats(parts)
            return loss, parts

        if isinstance(result, dict):
            parts = _parts_to_floats(result)
            if "total" in result and isinstance(result["total"], Tensor):
                loss = result["total"]
            else:
                # sum all tensor values as total
                total = 0.0
                for v in result.values():
                    if isinstance(v, Tensor):
                        total += float(v.detach().item())
                loss = torch.as_tensor(total, dtype=torch.float32, device=self.device)
            return loss, parts

        if isinstance(result, Tensor):
            return result, {"total": float(result.detach().item())}

        raise TypeError(f"Unsupported criterion return type: {type(result)}")


def _ensure_tensor(x: Union[Tensor, float], device: torch.device) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return torch.as_tensor(float(x), dtype=torch.float32, device=device)


def _parts_to_floats(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, Tensor):
            out[k] = float(v.detach().item())
        elif isinstance(v, (float, int)):
            out[k] = float(v)
        else:
            # ignore non-numerical parts for logging
            continue
    return out


def _merge_add(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    if not a:
        return dict(b)
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + float(v)
    return out


def _ts() -> str:
    return time.strftime("%H:%M:%S")