# src/engine/trainer.py
from __future__ import annotations
import math, os, logging
from typing import Dict, Optional, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0

def barrier():
    if is_dist():
        dist.barrier()

def ddp_module(model):
    return model.module if isinstance(model, DDP) else model

def reduce_tensor(t: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    if not is_dist():
        return t
    rt = t.clone()
    dist.all_reduce(rt, op=op)
    return rt

def _gpu_mem_str(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        mem = torch.cuda.memory_reserved() / (1024 ** 3)
        return f"{mem:.2f}G"
    return "0.00G"


class Trainer:
    """
    YOLO11-style trainer:
      - manual per-iteration warmup (optimizer updates)
      - epoch-stepped scheduler (LambdaLR), created after the first optimizer step
      - AMP + grad accumulation + DDP-safe
    Expects:
      - model(images) -> dict: {'det': det_maps, 'feats': feats}
      - criterion(det_maps, feats, batch, model=None, epoch=0) -> (loss, logs)
      - evaluator.evaluate(model, val_loader, device, max_images=None) -> metrics
    """

    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: GradScaler,
        device: str = "cuda",
        epochs: int = 100,
        use_amp: bool = True,
        scheduler: Optional[object] = None,  # may be a factory (callable) OR a built instance
        grad_accum: int = 1,
        max_grad_norm: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
        cfg: Optional[Any] = None,
        ckpt_dir: Optional[str] = None,
        warmup_epochs: float = 3.0,
        warmup_lr_init: float = 0.2,   # as a fraction of base_lr
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device
        self.epochs = int(epochs)
        self.use_amp = bool(use_amp)

        # --- scheduler: support lazy factory ---
        self._sched_factory = scheduler if callable(scheduler) else None
        self.scheduler = None if callable(scheduler) else scheduler
        self._sched_ready = self.scheduler is not None

        self.grad_accum = max(1, int(grad_accum))
        self.max_grad_norm = max_grad_norm if (max_grad_norm is None) else float(max_grad_norm)
        self.logger = logger or logging.getLogger("obbpose11.trainer")
        self.cfg = cfg
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.ckpt_dir = ckpt_dir or getattr(getattr(cfg, "train", object()), "save_dir", "runs/train/exp")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # warmup config
        self.warmup_epochs = float(warmup_epochs)
        self.warmup_lr_init = float(warmup_lr_init)
        self.global_update = 0  # counts optimizer updates across epochs

        trns = getattr(cfg, "train", None)
        self.sel_metric = getattr(trns, "selection_metric", "map50") if trns is not None else "map50"
        self.sel_mode = getattr(trns, "selection_mode", "max") if trns is not None else "max"
        self.best_metric = -float("inf") if self.sel_mode == "max" else float("inf")

        if self.rank == 0:
            per_gpu_batch = getattr(getattr(cfg, "train", object()), "batch", "NA")
            self.logger.info(f"[DDP] world={self.world_size} | per_gpu_batch={per_gpu_batch} | accum_steps={self.grad_accum}")

    # --------------------------------------------------------------------- #
    def fit(self, train_loader, val_loader, evaluator, train_sampler=None):
        device = self.device
        model = self.model
        scaler = self.scaler

        # eval cadence
        evcfg = getattr(self.cfg, "eval", None)
        eval_interval = int(getattr(evcfg, "interval", 1)) if evcfg else 1
        warmup_noeval = int(getattr(evcfg, "warmup_noeval_epochs", 0)) if evcfg else 0
        subset_frac = float(getattr(evcfg, "subset", 1.0)) if evcfg else 1.0
        fast_mode_default = bool(getattr(evcfg, "fast", False)) if evcfg else False
        full_every = int(getattr(evcfg, "full_every", 0)) if evcfg else 0

        # warmup in terms of optimizer updates (respecting grad accumulation)
        updates_per_epoch = math.ceil(len(train_loader) / self.grad_accum)
        warmup_updates = int(max(1, round(self.warmup_epochs * updates_per_epoch)))

        # cache base_lrs from the optimizer (set earlier in train.py)
        base_lrs = [pg.get("base_lr", pg["lr"]) for pg in self.optimizer.param_groups]

        for epoch in range(self.epochs):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            model.train()
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()

            agg = {"loss":0.0,"box":0.0,"obj":0.0,"ang":0.0,"kpt":0.0,"kc":0.0,"cls":0.0,"pos":0.0}
            n_steps = 0

            pbar = tqdm(total=len(train_loader), ncols=110, desc=f"{epoch+1}/{self.epochs}", disable=(self.rank != 0))
            self.optimizer.zero_grad(set_to_none=True)

            for it, batch in enumerate(train_loader):
                imgs = batch["image"].to(device, non_blocking=True)

                with autocast(device_type="cuda", enabled=self.use_amp):
                    outs = model(imgs)
                    if isinstance(outs, dict):
                        det_maps = outs.get("det") or outs.get("pred") or outs.get("yolo")
                        feats    = outs.get("feats") or outs.get("features")
                    else:
                        det_maps, feats = outs[0], outs[1]
                    if det_maps is None or feats is None:
                        raise RuntimeError("Model must return detection maps and feature maps (feats).")
                    model_for_loss = self.model.module if hasattr(self.model, "module") else self.model
                    loss, logs = self.criterion(det_maps, feats, batch, model=model_for_loss, epoch=epoch)

                loss = loss / self.grad_accum
                if self.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # optimizer update?
                do_step = ((it + 1) % self.grad_accum == 0) or ((it + 1) == len(train_loader))
                if do_step:
                    # --- per-iteration WARMUP (on optimizer updates) ---
                    if self.global_update < warmup_updates:
                        wu = (self.global_update + 1) / warmup_updates  # (0,1]
                        warm_factor = self.warmup_lr_init + (1.0 - self.warmup_lr_init) * wu
                        for (pg, base_lr) in zip(self.optimizer.param_groups, base_lrs):
                            pg["lr"] = base_lr * warm_factor

                    # clip before stepping
                    if self.max_grad_norm is not None:
                        if self.use_amp:
                            scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                    # OPTIMIZER STEP
                    if self.use_amp:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    # LAZY-CREATE EPOCH SCHEDULER AFTER FIRST STEP
                    if not self._sched_ready and self._sched_factory is not None:
                        # seed initial_lr for safety
                        for pg in self.optimizer.param_groups:
                            pg.setdefault("initial_lr", pg.get("base_lr", pg["lr"]))
                        self.scheduler = self._sched_factory(self.optimizer)
                        self._sched_ready = True

                    self.global_update += 1  # count optimizer updates

                # aggregate logs
                agg["loss"] += float(logs.get("loss", loss.item() * self.grad_accum))
                agg["box"]  += float(logs.get("loss_box", 0.0))
                agg["obj"]  += float(logs.get("loss_obj", 0.0))
                agg["ang"]  += float(logs.get("loss_ang", 0.0))
                agg["kpt"]  += float(logs.get("loss_kpt", 0.0))
                agg["kc"]   += float(logs.get("loss_kc", 0.0))
                agg["cls"]  += float(logs.get("loss_cls", 0.0))
                agg["pos"]  += float(logs.get("num_pos", 0.0))
                n_steps += 1

                if self.rank == 0:
                    lr_now = self.optimizer.param_groups[0]["lr"]
                    pbar.set_postfix_str(
                        f"GPU_mem {_gpu_mem_str(device)}  lr {lr_now:.3e}  "
                        f"box={logs.get('loss_box', 0.0):.3f}, obj={logs.get('loss_obj', 0.0):.3f}, "
                        f"ang={logs.get('loss_ang', 0.0):.3f}, kpt={logs.get('loss_kpt', 0.0):.3f}"
                    )
                    pbar.update(1)

            pbar.close()

            # average (reduce if DDP)
            for k in agg:
                tensor_val = torch.tensor(agg[k], device=device, dtype=torch.float32)
                tensor_val = reduce_tensor(tensor_val, op=dist.ReduceOp.SUM) if is_dist() else tensor_val
                agg[k] = float(tensor_val.item()) / self.world_size
            agg_avg = {k: v / max(1, n_steps) for k, v in agg.items()}

            # --- epoch scheduler step (once per epoch, AFTER many optimizer steps) ---
            if self.scheduler is not None:
                self.scheduler.step()

            if self.rank == 0:
                self.logger.info(
                    f"{epoch+1:>4}/{self.epochs:<4}    GPU_mem {_gpu_mem_str(device)}  "
                    f"box_loss {agg_avg['box']:.4f}  obj_loss {agg_avg['obj']:.4f}  "
                    f"ang_loss {agg_avg['ang']:.4f}  kpt_loss {agg_avg['kpt']:.4f}  kc_loss {agg_avg['kc']:.4f}  "
                    f"Pos {agg_avg['pos']:.1f}"
                )

            # ---------------------- evaluation cadence ---------------------- #
            do_eval = True
            if (epoch + 1) <= warmup_noeval:
                do_eval = False
            if eval_interval > 1 and ((epoch + 1) % eval_interval != 0):
                do_eval = False

            metrics = {}
            if do_eval and self.rank == 0:
                # (optional) fast/full toggles
                use_fast = fast_mode_default
                if full_every > 0 and ((epoch + 1) % full_every == 0):
                    use_fast = False

                with torch.inference_mode(), autocast(device_type="cuda", enabled=self.use_amp):
                    metrics = evaluator.evaluate(ddp_module(model), val_loader, device=device, max_images=None)

            # checkpointing
            if self.rank == 0:
                self._save_checkpoint(os.path.join(self.ckpt_dir, "last.pt"), metrics)
                sel_val = self._selection_value(metrics)
                if sel_val is not None:
                    is_better = (sel_val > self.best_metric) if self.sel_mode == "max" else (sel_val < self.best_metric)
                    if is_better:
                        self.best_metric = sel_val
                        self._save_checkpoint(os.path.join(self.ckpt_dir, "best.pt"), metrics)

            barrier()

    # --------------------------------------------------------------------- #
    def _selection_value(self, metrics: Dict[str, float]) -> Optional[float]:
        if not metrics:
            return None
        key = getattr(self, "sel_metric", "map50")
        aliases = {"map@.50":"map50","map50":"map50","map":"map","pck":"pck@0.05","pck@0.05":"pck@0.05"}
        return float(metrics.get(key, metrics.get(aliases.get(key,""), None))) if (key in metrics or aliases.get(key,"") in metrics) else None

    def _save_checkpoint(self, path: str, metrics: Dict[str, float]):
        state = {
            "model": ddp_module(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if isinstance(self.scaler, GradScaler) else {},
            "metrics": metrics,
            "cfg": self.cfg,
        }
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
