
# src/engine/trainer.py
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
from numbers import Number

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.utils.lr_utils import build_yolo_scheduler  # preferred package path

Tensor = torch.Tensor


# ------------------ small helpers ------------------
def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist() else 1


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)


def _stack_if_list(x: Union[Tensor, Sequence[Tensor]]) -> Tensor:
    if isinstance(x, Tensor):
        return x
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Empty image list in batch.")
        if all(isinstance(t, Tensor) for t in x):
            if x[0].ndim == 3:
                return torch.stack(x, dim=0)
            elif x[0].ndim == 4:
                return torch.stack(x, dim=0)
    raise TypeError(f"Unsupported image container type: {type(x)}")


def _extract_images_from_batch(batch: Dict[str, Any]) -> Tensor:
    for k in ("image", "images", "img", "imgs"):
        if k in batch and batch[k] is not None:
            return _stack_if_list(batch[k])
    if "inputs" in batch and batch["inputs"] is not None:
        return _stack_if_list(batch["inputs"])
    raise KeyError("Could not find image tensor in batch. Tried: image/images/img/imgs/inputs")


def _ensure_tensor(x: Union[Tensor, float], device: torch.device) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return torch.as_tensor(float(x), dtype=torch.float32, device=device)


def _parts_to_floats(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (d or {}).items():
        if isinstance(v, Tensor):
            out[k] = float(v.detach().item())
        elif isinstance(v, (float, int)):
            out[k] = float(v)
        elif isinstance(v, (list, tuple)) and len(v) and isinstance(v[0], (int, float)):
            out[k] = float(v[0])
    return out


def _merge_add(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = out.get(k, 0.0) + float(v)
    return out


def _metric_to_float(x):
    if isinstance(x, Tensor):
        return float(x.detach().item())
    try:
        return float(x)
    except Exception:
        return 0.0


def _ddp_mean_scalar(x: float, device: Optional[torch.device] = None) -> float:
    if not _is_dist():
        return float(x)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.tensor([float(x)], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t.item() / float(_world_size()))


def _clone_val_loader_rank0(loader: DataLoader) -> DataLoader:
    """If loader uses a DistributedSampler, rebuild a non-distributed loader for rank 0 eval."""
    sampler = getattr(loader, "sampler", None)
    looks_dist = False
    if DistributedSampler is not None and isinstance(sampler, DistributedSampler):
        looks_dist = True
    else:
        looks_dist = hasattr(sampler, "num_replicas") and hasattr(sampler, "rank")
    if not looks_dist:
        return loader

    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        collate_fn=loader.collate_fn,
        drop_last=False,
        persistent_workers=getattr(loader, "persistent_workers", False),
        prefetch_factor=getattr(loader, "prefetch_factor", None),
        multiprocessing_context=getattr(loader, "multiprocessing_context", None),
    )


# ------------------ Trainer ------------------
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        logger: Optional[Any] = None,
        cfg: Optional[Dict[str, Any]] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.cfg = cfg or {}

        self.rank = _rank()
        self.world_size = _world_size()
        self.is_main = (self.rank == 0)

        use_amp = bool(self.cfg.get("amp", False))
        if scaler is not None:
            self.scaler = scaler
        else:
            if use_amp and self.device.type == "cuda":
                try:
                    self.scaler = torch.amp.GradScaler("cuda")
                except TypeError:
                    self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            else:
                try:
                    self.scaler = torch.amp.GradScaler("cuda", enabled=False)
                except TypeError:
                    self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        self.epochs = int(self.cfg.get("epochs", 10))
        self.grad_clip_norm = float(self.cfg.get("grad_clip_norm", 0.0))
        self.eval_on_rank0_only = bool(self.cfg.get("eval_on_rank0_only", True))

    def fit(
            self,
            train_loader: Iterable[Dict[str, Any]],
            val_loader: Optional[Iterable[Dict[str, Any]]] = None,
            evaluator: Optional[Any] = None,
            train_sampler: Optional[Any] = None,
            log_interval: int = 0,
    ) -> Optional[float]:

        model = self.model.to(self.device)
        optimizer = self.optimizer
        criterion = self.criterion
        core_model = _unwrap_model(model)

        # --- YOLO-style LR schedule ---
        epochs = self.epochs
        iters_per_epoch = len(train_loader) if hasattr(train_loader, "__len__") else None
        lr0 = float(self.cfg.get("lr0", 0.01))
        lrf = float(self.cfg.get("lrf", 0.01))
        warmup_epochs = float(self.cfg.get("warmup_epochs", 3.0))
        momentum = float(self.cfg.get("momentum", 0.937))
        warmup_momentum = float(self.cfg.get("warmup_momentum", 0.8))

        yolo_sched, warmup_state, warmup_step = build_yolo_scheduler(
            optimizer,
            epochs=epochs,
            lr0=lr0,
            lrf=lrf,
            warmup_epochs=warmup_epochs,
            momentum=momentum,
            warmup_momentum=warmup_momentum,
            iters_per_epoch=iters_per_epoch,
        )
        self.scheduler = yolo_sched

        if self.is_main:
            print(f"[LR] lr0={lr0} lrf={lrf} warmup_epochs={warmup_epochs} "
                  f"momentum={momentum} warmup_momentum={warmup_momentum}", flush=True)

        eval_every = int(self.cfg.get("eval_interval", 1))
        warmup_noeval = int(self.cfg.get("warmup_noeval", 0))
        select_name = str(self.cfg.get("select", "mAP50"))
        mode = str(self.cfg.get("mode", "max")).lower()
        best_metric: Optional[float] = None

        global_iter = 0
        for epoch in range(1, epochs + 1):
            model.train()
            if _is_dist() and train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            running_total = 0.0
            parts_accum: Dict[str, float] = {}
            steps = 0

            for step, batch in enumerate(train_loader, start=1):
                imgs = _extract_images_from_batch(batch).to(self.device, non_blocking=True)

                # per-iteration warmup
                warmup_step(global_iter)

                # forward in AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.scaler.is_enabled()):
                    outputs = model(imgs)

                # compute loss in fp32 for stability
                with torch.amp.autocast('cuda', enabled=False):
                    loss, parts = self._call_criterion(criterion, outputs, batch, epoch)

                # NaN / Inf guard
                if not torch.isfinite(loss):
                    if self.is_main:
                        p = _parts_to_floats(parts)
                        print(f"[NAN GUARD] non-finite loss at epoch {epoch}, step {step}. parts={p}", flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    global_iter += 1
                    continue

                # grad clip (unscaled if using GradScaler)
                if self.grad_clip_norm and self.grad_clip_norm > 0:
                    if self.scaler and self.scaler.is_enabled():
                        self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)

                # step
                if self.scaler and self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                running_total += float(loss.detach().item())
                parts_accum = _merge_add(parts_accum, parts)
                steps += 1

                if log_interval > 0 and (step % log_interval == 0) and self.is_main:
                    avg_loss = running_total / max(1, steps)
                    print(f"[rank{self.rank}] epoch {epoch} step {step}: loss={avg_loss:.4f}", flush=True)

                global_iter += 1

            epoch_loss = running_total / max(1, steps)
            epoch_loss_avg = _ddp_mean_scalar(epoch_loss, device=self.device)
            if self.is_main:
                print(f"[{_ts()}] Epoch {epoch}: loss={epoch_loss_avg:.4f}", flush=True)

            if self.scheduler is not None:
                self.scheduler.step()
                if self.is_main and hasattr(optimizer, "param_groups"):
                    cur_lr = optimizer.param_groups[0]["lr"]
                    print(f"[LR] epoch {epoch} -> lr={cur_lr:.6g}", flush=True)

            # ---- Evaluation ----
            do_eval = (evaluator is not None) and (val_loader is not None) and (eval_every > 0) and (
                        epoch % eval_every == 0)
            if warmup_noeval > 0 and epoch <= warmup_noeval:
                do_eval = False

            if do_eval:
                metrics = {}
                if _is_dist() and self.eval_on_rank0_only:
                    if self.rank == 0:
                        vloader = _clone_val_loader_rank0(val_loader)
                        core_model.eval()
                        with torch.inference_mode():
                            metrics = evaluator.evaluate(model=core_model, val_loader=vloader, device=self.device)
                        core_model.train()
                    obj = [metrics]
                    if _is_dist():
                        dist.broadcast_object_list(obj, src=0)
                    metrics = obj[0]
                else:
                    core_model.eval()
                    with torch.inference_mode():
                        metrics = evaluator.evaluate(model=core_model, val_loader=val_loader, device=self.device)
                    core_model.train()

                if self.is_main and metrics:
                    sel = _metric_to_float(metrics.get(select_name, 0.0))
                    print(f"[{_ts()}] Eval: {select_name}={sel:.6f} (mode={mode})", flush=True)
                    if best_metric is None:
                        best_metric = sel
                    else:
                        if (mode == "max" and sel > best_metric) or (mode == "min" and sel < best_metric):
                            best_metric = sel
            else:
                if self.is_main and epoch == 1:
                    print("[WARN] No evaluator or val_loader provided or eval skipped; metric will not be computed.",
                          flush=True)

        return best_metric

    # ------------------ criterion adapter ------------------
    def _call_criterion(self, criterion: nn.Module, outputs: Any, batch: Dict[str, Any], epoch: int):
        if isinstance(outputs, (list, tuple)):
            if len(outputs) == 2 and hasattr(criterion, "forward"):
                det_maps, feats = outputs
                try:
                    result = criterion(det_maps, feats, batch, model=_unwrap_model(self.model), epoch=epoch)
                except TypeError:
                    result = criterion(det_maps, feats, batch)
            else:
                try:
                    result = criterion(outputs, batch)
                except TypeError:
                    result = criterion(outputs)
        else:
            try:
                result = criterion(outputs, batch)
            except TypeError:
                result = criterion(outputs)

        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], (Tensor, float, int)):
            loss, parts = result
            parts = _parts_to_floats(parts)
            if isinstance(loss, Tensor):
                return loss, parts
            else:
                return _ensure_tensor(loss, device=self.device), parts

        if isinstance(result, dict):
            parts = _parts_to_floats(result)
            total = 0.0
            for v in result.values():
                if isinstance(v, Tensor):
                    total += float(v.detach().item())
            loss = torch.as_tensor(total, dtype=torch.float32, device=self.device)
            return loss, parts

        if isinstance(result, Tensor):
            return result, {"total": float(result.detach().item())}

        raise TypeError(f"Unsupported criterion return type: {type(result)}")