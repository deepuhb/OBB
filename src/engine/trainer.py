# src/engine/trainer.py

import os, math, logging
from contextlib import nullcontext
from typing import Any, Dict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


# --------------------------- small helpers ---------------------------

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def rank() -> int:
    return dist.get_rank() if is_dist() else 0

def barrier():
    if is_dist():
        dist.barrier()

def ddp_module(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, DDP) else m

def _gpu_mem_str(device: str) -> str:
    if isinstance(device, str) and device.startswith("cuda"):
        try:
            return f"{torch.cuda.max_memory_allocated() / (1024**3):.2f}G"
        except Exception:
            return "cuda"
    return "CPU"

def _to_device(obj: Any, device: str) -> Any:
    """Recursively move tensors to device."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        typ = list if isinstance(obj, list) else tuple
        return typ(_to_device(v, device) for v in obj)
    return obj


# --------------------------- main trainer ---------------------------

class Trainer:
    """
    Trainer with:
      - DDP + no_sync for grad accumulation
      - AMP (new and old autocast)
      - warmup per update + optional cosine epoch scheduler
      - dynamic loss balancer hook (multi_noise)
      - rank0 evaluation on a non-distributed loader
      - metric-based checkpointing (best/last)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        *,
        cfg=None,
        scaler=None,
        device: str = None,
        epochs: int = 100,
        grad_accum: int = 1,
        logger: logging.Logger = None,
        ckpt_dir: str = None,
        use_amp: bool = True,
        max_grad_norm: float | None = None,
        warmup_epochs: float = 0.0,
        warmup_lr_init: float = 0.2,
        cosine_factor=None,           # callable(epoch_idx)->scale OR None
        multi_noise=None,             # optional dynamic loss balancer
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.cfg = cfg
        self.scaler = scaler
        self.use_amp = bool(use_amp)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = int(epochs)
        self.grad_accum = max(1, int(grad_accum))
        self.logger = logger or logging.getLogger(__name__)
        self.max_grad_norm = max_grad_norm
        self.warmup_epochs = float(warmup_epochs)
        self.warmup_lr_init = float(warmup_lr_init)
        self.cosine_factor = cosine_factor
        self.multi_noise = multi_noise
        self.use_multi_noise = multi_noise is not None

        # distributed
        self.rank = rank()
        self.world_size = world_size()

        # lrs bookkeeping
        self.base_lrs = [pg.get("initial_lr", pg["lr"]) for pg in self.optimizer.param_groups]

        # ckpt / selection config
        train_cfg = getattr(cfg, "train", None)
        self.ckpt_dir = ckpt_dir or (getattr(train_cfg, "save_dir", None) or "runs/exp")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        eval_cfg = getattr(cfg, "eval", None)
        self.eval_interval = int(getattr(eval_cfg, "interval", 1)) if eval_cfg else 1
        self.warmup_noeval = int(getattr(eval_cfg, "warmup_noeval_epochs", 0)) if eval_cfg else 0
        self.sel_metric = (getattr(eval_cfg, "select", None) or "mAP50")
        self.sel_mode = (getattr(eval_cfg, "select_mode", None) or "max")
        self.best_metric = float("-inf") if self.sel_mode == "max" else float("inf")
        self.logger.info(f"[EVAL cfg] interval={self.eval_interval}  warmup_noeval={self.warmup_noeval}  "
                         f"select='{self.sel_metric}' mode='{self.sel_mode}'")

        # global update counter (for warmup per update)
        self.global_update = 0

    # ---------- autocast compatibility (new & old AMP APIs) ----------

    def _autocast(self):
        """
        Returns a context manager:
        - new API: torch.amp.autocast('cuda', enabled=True)
        - old API fallback: torch.cuda.amp.autocast(enabled=True)
        - CPU / disabled AMP: nullcontext()
        """
        if not self.use_amp or not str(self.device).startswith("cuda"):
            return nullcontext()
        try:
            # PyTorch >= 2.1
            import torch.amp as ta
            return ta.autocast("cuda", enabled=True)
        except Exception:
            # Older API
            from torch.cuda.amp import autocast as old_autocast
            return old_autocast(enabled=True)

    def _rank0_eval_loader(self, val_loader):
        """Build a non-distributed eval loader over the full dataset for rank 0."""
        ds = val_loader.dataset
        return DataLoader(
            ds,
            batch_size=val_loader.batch_size,  # per-GPU batch size is fine here
            shuffle=False,
            sampler=None,  # IMPORTANT: no DistributedSampler
            drop_last=False,
            num_workers=getattr(val_loader, "num_workers", 0),
            pin_memory=getattr(val_loader, "pin_memory", True),
            collate_fn=val_loader.collate_fn,
            persistent_workers=getattr(val_loader, "persistent_workers", False),
        )

    # --------------------- public API ---------------------

    def fit(self, train_loader, val_loader, evaluator, train_sampler=None):
        device = self.device
        model = self.model
        scaler = self.scaler

        updates_per_epoch = max(1, math.ceil(len(train_loader) / self.grad_accum))
        warmup_updates = int(max(1, round(self.warmup_epochs * updates_per_epoch)))

        for epoch in range(self.epochs):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            model.train()
            if str(device).startswith("cuda"):
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

            # epoch aggregates
            agg = {"loss": 0.0, "box": 0.0, "obj": 0.0, "ang": 0.0, "kpt": 0.0, "kc": 0.0, "cls": 0.0, "pos": 0.0}
            n_steps = 0

            pbar = tqdm(
                total=len(train_loader),
                ncols=110,
                desc=f"{epoch+1}/{self.epochs}",
                disable=(self.rank != 0),
            )

            self.optimizer.zero_grad(set_to_none=True)

            for it, batch in enumerate(train_loader):
                batch = _to_device(batch, device)

                # ---- robust image extraction: image/images/imgs/img ----
                imgs = None
                if isinstance(batch, dict):
                    for k in ("image", "images", "imgs", "img"):
                        if k in batch:
                            imgs = batch[k]
                            break
                if imgs is None:
                    raise RuntimeError("Batch must contain key 'image' (or 'images'/'imgs'/'img') with (B,C,H,W).")
                if not torch.is_tensor(imgs):
                    imgs = torch.stack(imgs, dim=0)

                # Accumulation: only sync on step boundaries
                do_sync = ((it + 1) % self.grad_accum == 0) or ((it + 1) == len(train_loader))
                sync_ctx = nullcontext()
                if isinstance(self.model, DDP) and not do_sync:
                    sync_ctx = self.model.no_sync()

                with sync_ctx:
                    with self._autocast():
                        outs = model(imgs)

                        # model may return tuple or dict; prefer dict with det/feats
                        if isinstance(outs, dict):
                            det_maps = outs.get("det") or outs.get("pred") or outs.get("yolo")
                            feats    = outs.get("feats") or outs.get("features")
                            if det_maps is None or feats is None:
                                raise RuntimeError(f"Model outputs missing 'det'/'feats'. Got keys: {list(outs.keys())}")
                        else:
                            if not isinstance(outs, (list, tuple)) or len(outs) < 2:
                                raise RuntimeError("Model must return (det_maps, feats) or dict with keys.")
                            det_maps, feats = outs[0], outs[1]

                        model_for_loss = ddp_module(self.model)

                        # ---- loss: kwargs first, then legacy positional fallback ----
                        try:
                            loss, logs = self.criterion(
                                det_maps=det_maps,
                                feats=feats,
                                batch=batch,
                                model=model_for_loss,
                                epoch=epoch,
                            )
                        except TypeError:
                            loss, logs = self.criterion(det_maps, feats, batch, model=model_for_loss, epoch=epoch)

                        # optional dynamic loss balancing (expects per-task raw losses)
                        if self.multi_noise is not None:
                            raw_keys = ("box_loss_raw", "obj_loss_raw", "ang_loss_raw", "kpt_loss_raw", "kc_loss_raw", "cls_loss_raw")
                            raw_vals = []
                            for k in raw_keys:
                                v = logs.get(k, None) if isinstance(logs, dict) else None
                                if v is None:
                                    continue
                                if not torch.is_tensor(v):
                                    v = torch.as_tensor(v, device=device, dtype=torch.float32)
                                raw_vals.append(v)
                            if raw_vals:
                                loss = self.multi_noise(raw_vals)

                        # scale by accumulation
                        loss = loss / self.grad_accum

                    if self.use_amp and scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                if do_sync:
                    # warmup per update
                    if self.global_update < warmup_updates:
                        u = (self.global_update + 1) / warmup_updates
                        warm_factor = self.warmup_lr_init + (1.0 - self.warmup_lr_init) * u
                        for (pg, base_lr) in zip(self.optimizer.param_groups, self.base_lrs):
                            pg["lr"] = base_lr * warm_factor

                    # clip & step
                    if self.max_grad_norm is not None:
                        if self.use_amp and scaler is not None:
                            scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                    if self.use_amp and scaler is not None:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_update += 1

                # aggregate logs
                logs = logs or {}
                agg["loss"] += float(logs.get("loss", (loss.item() * self.grad_accum)))
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
                    # cast to float for safety
                    box_s = float(logs.get('loss_box', 0.0))
                    obj_s = float(logs.get('loss_obj', 0.0))
                    ang_s = float(logs.get('loss_ang', 0.0))
                    kpt_s = float(logs.get('loss_kpt', 0.0))
                    cls_s = float(logs.get('loss_cls', 0.0)) if 'loss_cls' in logs else 0.0
                    pbar.set_postfix_str(
                        f"GPU_mem {_gpu_mem_str(device)}  lr {lr_now:.3e}  "
                        f"box={box_s:.3f}, obj={obj_s:.3f}, ang={ang_s:.3f}, kpt={kpt_s:.3f}"
                        + (f", cls={cls_s:.3f}" if 'loss_cls' in logs else "")
                    )
                    pbar.update(1)

            pbar.close()

            # reduce across ranks then average per step
            for k in agg:
                t = torch.tensor(agg[k], device=device, dtype=torch.float32)
                if is_dist():
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                agg[k] = float(t.item()) / self.world_size
            agg_avg = {k: v / max(1, n_steps) for k, v in agg.items()}

            # cosine per-epoch after warmup
            if self.cosine_factor is not None and self.global_update >= warmup_updates:
                factor = float(self.cosine_factor(epoch + 1))
                for (pg, base_lr) in zip(self.optimizer.param_groups, self.base_lrs):
                    pg["lr"] = base_lr * factor

            if self.rank == 0:
                self.logger.info(
                    f"{epoch+1:>4}/{self.epochs:<4}    GPU_mem {_gpu_mem_str(device)}  "
                    f"box_loss {agg_avg['box']:.4f}  obj_loss {agg_avg['obj']:.4f}  "
                    f"ang_loss {agg_avg['ang']:.4f}  kpt_loss {agg_avg['kpt']:.4f}  "
                    f"kc_loss {agg_avg['kc']:.4f}  cls_loss {agg_avg['cls']:.4f}  Pos {agg_avg['pos']:.1f}"
                )

            # evaluation cadence (rank0)
            do_eval = True
            if (epoch + 1) <= self.warmup_noeval:
                do_eval = False
            if self.eval_interval > 1 and ((epoch + 1) % self.eval_interval != 0):
                do_eval = False

            # --- evaluation (rank0 only, every epoch according to cadence) ---
            metrics: Dict[str, Any] = {}
            if evaluator is not None and do_eval:
                if self.rank == 0:
                    # build a non-distributed, full-dataset loader for eval
                    if isinstance(getattr(val_loader, "sampler", None), DistributedSampler):
                        eval_loader = DataLoader(
                            val_loader.dataset,
                            batch_size=val_loader.batch_size,
                            shuffle=False,
                            sampler=None,  # IMPORTANT: no DistributedSampler
                            drop_last=False,
                            num_workers=getattr(val_loader, "num_workers", 0),
                            pin_memory=getattr(val_loader, "pin_memory", True),
                            collate_fn=val_loader.collate_fn,
                            persistent_workers=getattr(val_loader, "persistent_workers", False),
                        )
                    else:
                        eval_loader = val_loader  # already non-distributed

                    with torch.inference_mode():
                        with self._autocast():
                            metrics = evaluator.evaluate(ddp_module(model), eval_loader, device=device, max_images=None)

                    if isinstance(metrics, dict) and metrics:
                        self._log_eval(metrics)
                    else:
                        (self.logger or logging.getLogger("obbpose11")).warning("[EVAL] evaluator returned no metrics")

                    # checkpoints (rank0)
                    self._save_checkpoint(os.path.join(self.ckpt_dir, "last.pt"), metrics)
                    sel_val = self._selection_value(metrics)
                    if sel_val is not None:
                        is_better = (sel_val > self.best_metric) if self.sel_mode == "max" else (sel_val < self.best_metric)
                        if is_better:
                            self.best_metric = sel_val
                            self._save_checkpoint(os.path.join(self.ckpt_dir, "best.pt"), metrics)

                # keep other ranks in lock-step during eval epoch
                if dist.is_initialized():
                    dist.barrier()

    # --------------------- utils ---------------------

    def _selection_value(self, metrics: Dict[str, Any] | None):
        if not isinstance(metrics, dict) or not metrics:
            return None
        key = self.sel_metric
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except Exception:
                return None
        return None

    def _save_checkpoint(self, path: str, metrics: Dict[str, Any] | None):
        state = {
            "model": ddp_module(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else {},
            "scaler": self.scaler.state_dict() if self.scaler is not None else {},
            "metrics": metrics or {},
            "cfg": self.cfg,
        }
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
        (self.logger or logging.getLogger("obbpose11")).info(f"[ckpt] saved: {path}")

    def _log_eval(self, metrics: dict):
        keys = [
            ("images", None),
            ("mAP50", ".6f"),
            ("mAP", ".6f"),
            ("pck@0.05", ".6f"),
            ("pck_any@0.05", ".6f"),
            ("tps", None),
            ("pred_per_img", ".2f"),
            ("recall@0.1", ".2f"),
            ("recall@0.3", ".2f"),
            ("recall@0.5", ".2f"),
            ("best_iou", ".3f"),
        ]
        parts = []
        for k, fmt in keys:
            if k in metrics and metrics[k] is not None:
                v = metrics[k]
                try:
                    v = float(v)
                except Exception:
                    pass
                parts.append(f"{k} {v:{fmt}}" if fmt else f"{k} {v}")
        msg = "  ".join(parts) if parts else str(metrics)
        (self.logger or logging.getLogger("obbpose11")).info(f"[EVAL] {msg}")
