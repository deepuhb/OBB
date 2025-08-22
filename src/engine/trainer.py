
# src/engine/trainer.py
from __future__ import annotations

import math
import time
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None


def _is_dist():
    return (dist is not None) and dist.is_available() and dist.is_initialized()


def _get_rank():
    if _is_dist():
        return dist.get_rank()
    return 0


def _is_main():
    return _get_rank() == 0


def _to_device_sample(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move tensors in the batch dict to device. Keep lists of per-sample tensors intact."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, (list, tuple)):
            moved = []
            for x in v:
                if torch.is_tensor(x):
                    moved.append(x.to(device, non_blocking=True))
                else:
                    moved.append(x)
            out[k] = moved
        else:
            out[k] = v
    return out


@dataclass
class TrainCfg:
    epochs: int = 100
    accum_steps: int = 1
    amp: bool = True
    log_interval: int = 50
    grad_clip: Optional[float] = None  # e.g., 10.0
    eval_interval: int = 1
    warmup_noeval: int = 0  # epochs to skip eval at start


@dataclass
class EvalCfg:
    select: str = "mAP50"
    mode: str = "max"  # "max" or "min"


class Trainer:
    """
    Trainer with YOLO11-style AMP + grad accumulation + eval/checkpointing.
    Expects: evaluator.evaluate(model, val_loader, device=...) -> dict[str, float]
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        cfg: Any,
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Sub-configs with sane defaults
        self.tc = getattr(cfg, "train", TrainCfg())
        self.ec = getattr(cfg, "eval",  EvalCfg())
        for k, v in dict(epochs=100, accum_steps=1, amp=True, log_interval=50,
                         grad_clip=None, eval_interval=1, warmup_noeval=0).items():
            if not hasattr(self.tc, k): setattr(self.tc, k, v)
        for k, v in dict(select="mAP50", mode="max").items():
            if not hasattr(self.ec, k): setattr(self.ec, k, v)

        # AMP
        self.use_amp = bool(self.tc.amp)
        self.scaler = GradScaler(device="cuda", enabled=self.use_amp)

        self.logger = logger

        # Optional checkpoint helper
        self._ckpt = None
        try:
            from src.engine import checkpoint as _ckpt
            self._ckpt = _ckpt
        except Exception:
            self._ckpt = None

    def _log(self, msg: str):
        if _is_main():
            ts = time.strftime("[%H:%M:%S] ")
            if self.logger is not None:
                try:
                    self.logger.info(msg)
                except Exception:
                    print(ts + msg, flush=True)
            else:
                print(ts + msg, flush=True)

    def _reduce_scalar(self, x: torch.Tensor) -> float:
        """All-reduce a scalar (mean) across ranks if DDP, else return float(x)."""
        if not torch.is_tensor(x):
            x = torch.tensor(float(x), device=self.device)
        if _is_dist():
            xt = x.detach().clone()
            dist.all_reduce(xt, op=dist.ReduceOp.AVG)
            return float(xt.item())
        else:
            return float(x.item())

    def _maybe_clip(self):
        if self.tc.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip)

    def _best_is_better(self, current: float, best: Optional[float]) -> bool:
        if best is None:
            return True
        return (current > best) if str(self.ec.mode).lower() == "max" else (current < best)

    def _save_ckpt(self, path: str, epoch: int, metrics: Dict[str, float], best_metric: Optional[float] = None):
        tosave = {
            "epoch": epoch,
            "model": (self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "metrics": metrics,
            "best": best_metric,
        }
        if self._ckpt is not None and hasattr(self._ckpt, "save_checkpoint"):
            try:
                self._ckpt.save_checkpoint(tosave, path)
                return
            except Exception:
                pass
        torch.save(tosave, path)

    @staticmethod
    def _unwrap_ddp(m):
        return m.module if hasattr(m, "module") else m

    def _call_criterion(self, criterion, outputs, batch, device, epoch: int):
        """
        Adapt to different loss signatures.
        Supports any (subset) of:
          (outputs, batch), (outputs, targets), (outputs, targets, batch),
          (det_maps, feats, batch, model, epoch, device=...)
        """
        # Prepare unpacked outputs if dict-like
        det_maps = outputs.get("det") if isinstance(outputs, dict) else None
        feats = outputs.get("feats") if isinstance(outputs, dict) else None
        kpt_maps = outputs.get("kpt_maps") if isinstance(outputs, dict) else None

        # Targets from batch if present, otherwise derive as "everything but image"
        targets = batch.get("targets", None) if isinstance(batch, dict) else None
        if targets is None and isinstance(batch, dict):
            targets = {k: v for k, v in batch.items() if k != "image"}

        sig = None
        try:
            sig = inspect.signature(criterion.forward)
        except Exception:
            pass

        core_model = self._unwrap_ddp(self.model)

        if sig is not None:
            params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
            names = [p.name for p in params][1:]  # drop 'self'

            values = {}
            for name in names:
                low = name.lower()
                if low in ("det_maps", "det", "preds_det", "maps"):
                    values[name] = det_maps if det_maps is not None else outputs
                elif low in ("feats", "features"):
                    values[name] = feats
                elif low in ("kpt_maps", "kpm"):
                    values[name] = kpt_maps
                elif low in ("outputs", "out", "preds", "pred", "y_pred", "logits"):
                    values[name] = outputs
                elif low in ("batch", "data", "samples"):
                    values[name] = batch
                elif low in ("targets", "target", "gt", "gts", "labels_targets"):
                    values[name] = targets
                elif low == "model":
                    values[name] = self.model
                elif low == "epoch":
                    values[name] = epoch
                elif low == "device":
                    values[name] = device
                elif low == "model":
                    values[name] = core_model

            # Try keyword call first
            try:
                return criterion(**values)
            except TypeError:
                pass

        # Fallbacks with positional mapping
        # 1) Modern loss signature: (det_maps, feats, batch, model=None, epoch=0)
        if det_maps is not None and feats is not None:
            try:
                return criterion(det_maps, feats, batch, model=core_model, epoch=epoch)
            except TypeError:
                try:
                    return criterion(det_maps, feats, batch)
                except TypeError:
                    pass

        # 2) Classic two-arg: (outputs, batch)
        try:
            return criterion(outputs, batch)
        except TypeError:
            pass

        # 3) (outputs, targets)
        try:
            return criterion(outputs, targets)
        except TypeError:
            pass

        # 4) (outputs) last resort
        return criterion(outputs)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        evaluator: Optional[Any],
        train_sampler: Optional[Any] = None,
    ):
        device = self.device
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler

        epochs = int(self.tc.epochs)
        accum_steps = max(1, int(self.tc.accum_steps))
        core_model = self._unwrap_ddp(model)

        self._log(f"[EVAL cfg] interval={self.tc.eval_interval}  warmup_noeval={self.tc.warmup_noeval}  select='{self.ec.select}' mode='{self.ec.mode}'")

        best_metric = None

        for epoch in range(1, epochs + 1):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                try:
                    train_sampler.set_epoch(epoch)
                except Exception:
                    pass

            model.train()
            epoch_loss = 0.0
            num_batches = 0
            loss_components: Dict[str, float] = {}

            pbar = None
            if _is_main():
                try:
                    from tqdm import tqdm
                    pbar = tqdm(total=len(train_loader), desc=f"{epoch}/{epochs}", leave=True, ncols=100)
                except Exception:
                    pbar = None

            optimizer.zero_grad(set_to_none=True)

            for bi, batch in enumerate(train_loader):
                batch = _to_device_sample(batch, device)
                imgs = batch["image"]  # [B,3,H,W]

                with autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                    outputs = model(imgs)
                    loss_obj = self._call_criterion(criterion, outputs, batch, device, epoch)
                    if isinstance(loss_obj, tuple):
                        loss, logs = loss_obj
                    elif isinstance(loss_obj, dict):
                        loss = sum(loss_obj.values())
                        logs = loss_obj
                    else:
                        loss = loss_obj
                        logs = {}

                    if torch.isnan(loss) or torch.isinf(loss):
                        self._log(f"Warning: NaN/Inf loss at epoch {epoch} iter {bi}, skipping step.")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                # accumulate component losses for logging
                if isinstance(logs, dict):
                    for k, v in logs.items():
                        try:
                            loss_components[k] = loss_components.get(k, 0.0) + float(v.detach().item())
                        except Exception:
                            pass

                # Gradient accumulation
                loss_scaled = loss / accum_steps
                self.scaler.scale(loss_scaled).backward()

                if (bi + 1) % accum_steps == 0:
                    if self.tc.grad_clip is not None:
                        self.scaler.unscale_(optimizer)
                        self._maybe_clip()

                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    if scheduler is not None and getattr(scheduler, "step_on_iter", False):
                        scheduler.step()

                epoch_loss += float(loss.detach().item())
                num_batches += 1
                if pbar is not None:
                    # show the last few keys
                    if loss_components:
                        tail = list(loss_components.items())[-3:]
                        tail_str = " ".join([f"{k}={v/max(1,num_batches):.3f}" for k, v in tail])
                        pbar.set_postfix({"loss": epoch_loss / max(1, num_batches), "comp": tail_str})
                    else:
                        pbar.set_postfix({"loss": epoch_loss / max(1, num_batches)})
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

            # Step LR per-epoch if that's your policy
            if scheduler is not None and not getattr(scheduler, "step_on_iter", False):
                try:
                    scheduler.step()
                except Exception:
                    pass

            mean_epoch_loss = (epoch_loss / max(1, num_batches)) if num_batches > 0 else float("nan")
            self._log(f"Epoch {epoch}: loss={mean_epoch_loss:.4f}")
            if len(loss_components):
                nsteps = max(1, num_batches)
                comps = " ".join([f"{k}={v/nsteps:.4f}" for k, v in loss_components.items()])
                self._log(f"Epoch {epoch} components: {comps}")

            # ---- Eval ----
            do_eval = (evaluator is not None) and (val_loader is not None)
            if do_eval and (epoch > self.tc.warmup_noeval) and ((epoch % self.tc.eval_interval) == 0):
                model.eval()
                with torch.no_grad():
                    with autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                        try:
                            metrics = evaluator.evaluate(core_model, val_loader, device=device)
                            if not isinstance(metrics, dict):
                                metrics = {"metric": float(metrics)}
                        except Exception as e:
                            self._log(f"Eval error at epoch {epoch}: {e}")
                            metrics = {}

                sel_key = str(self.ec.select)
                cur_val = float(metrics.get(sel_key, float('nan')))
                if math.isnan(cur_val):
                    self._log(f"Eval: key '{sel_key}' missing; metrics={metrics}")
                else:
                    self._log(f"Eval: {sel_key}={cur_val:.6f}  (mode={self.ec.mode})")

                # Save last & best
                if _is_main():
                    self._save_ckpt("runs/last.pt", epoch, metrics, best_metric)
                    if not math.isnan(cur_val) and self._best_is_better(cur_val, best_metric):
                        best_metric = cur_val
                        self._save_ckpt("runs/best.pt", epoch, metrics, best_metric)
                        self._log(f"New best {sel_key}={best_metric:.6f} — saved runs/best.pt")
            else:
                if _is_main():
                    self._save_ckpt("runs/last.pt", epoch, {"loss": mean_epoch_loss}, best_metric)

        self._log("Training complete.")