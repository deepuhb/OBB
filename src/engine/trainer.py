# src/engine/trainer.py
from __future__ import annotations
import math
import os
import time
import copy
from typing import Dict, Optional, Tuple, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.amp import autocast

from tqdm import tqdm
import logging


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
    # unwrap DDP if needed
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
    Generic trainer for YOLO11 OBB+top-down 1-keypoint.

    Expects:
      - model(images) -> dict with keys: 'det' (list of maps), 'feats' (neck feats)
      - criterion(det_maps, batch) -> (loss, logs_dict)
        logs_dict may include keys: loss, loss_box, loss_obj, loss_ang, loss_kpt, loss_cls, num_pos, etc.
      - evaluator.evaluate(model, val_loader, device, max_images=None) -> metrics dict

    Checkpointing:
      - Saves 'last.pt' every epoch and 'best.pt' when selection metric improves.
      - Selection metric default: 'map50' (configurable).
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
        scheduler: Optional[Any] = None,
        grad_accum: int = 1,
        max_grad_norm: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
        cfg: Optional[Any] = None,
        ckpt_dir: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device
        self.epochs = int(epochs)
        self.use_amp = bool(use_amp)
        self.scheduler = scheduler
        self.grad_accum = max(1, int(grad_accum))
        self.max_grad_norm = max_grad_norm if (max_grad_norm is None) else float(max_grad_norm)
        self.logger = logger or logging.getLogger("obbpose11.trainer")
        self.cfg = cfg
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.ckpt_dir = ckpt_dir or getattr(getattr(cfg, "train", object()), "save_dir", "runs/train/exp")

        os.makedirs(self.ckpt_dir, exist_ok=True)

        # selection metric & mode
        trns = getattr(cfg, "train", None)
        self.sel_metric = getattr(trns, "selection_metric", "map50") if trns is not None else "map50"
        self.sel_mode = getattr(trns, "selection_mode", "max") if trns is not None else "max"
        self.best_metric = -float("inf") if self.sel_mode == "max" else float("inf")

        # logging header
        if self.rank == 0:
            self.logger.info(
                f"[DDP] world={self.world_size} | per_gpu_batch={getattr(cfg.train, 'batch', 'NA')}"
                f" | accum_steps={self.grad_accum}"
            )

    # --------------------------------------------------------------------- #
    # core loop
    # --------------------------------------------------------------------- #
    def fit(self, train_loader, val_loader, evaluator, train_sampler=None):
        device = self.device
        model = self.model
        scaler = self.scaler

        # evaluation pacing knobs from cfg.eval
        evcfg = getattr(self.cfg, "eval", None)
        eval_interval = int(getattr(evcfg, "interval", 1)) if evcfg else 1
        warmup_noeval = int(getattr(evcfg, "warmup_noeval_epochs", 0)) if evcfg else 0
        subset_frac = float(getattr(evcfg, "subset", 1.0)) if evcfg else 1.0
        fast_mode_default = bool(getattr(evcfg, "fast", False)) if evcfg else False
        full_every = int(getattr(evcfg, "full_every", 0)) if evcfg else 0

        for epoch in range(self.epochs):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            model.train()
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()

            # epoch aggregations
            agg = {
                "loss": 0.0, "box": 0.0, "obj": 0.0, "ang": 0.0, "kpt": 0.0, "kc": 0.0, "cls": 0.0, "pos": 0.0
            }
            n_steps = 0
            t0 = time.time()

            pbar = None
            if self.rank == 0:
                pbar = tqdm(total=len(train_loader), ncols=100, desc=f"{epoch+1}/{self.epochs}")

            self.optimizer.zero_grad(set_to_none=True)

            for it, batch in enumerate(train_loader):
                imgs = batch["image"].to(device, non_blocking=True)

                # forward + loss
                with autocast(device_type="cuda", enabled=self.use_amp):
                    outs = model(imgs)
                    loss, logs = self.criterion(outs["det"], batch)

                # scale loss by grad_accum
                loss = loss / self.grad_accum

                if self.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # step if needed
                if (it + 1) % self.grad_accum == 0:
                    if self.max_grad_norm is not None:
                        if self.use_amp:
                            scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

                    if self.use_amp:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                # aggregate logs (defensive)
                agg["loss"] += float(logs.get("loss", loss.item() * self.grad_accum))
                agg["box"]  += float(logs.get("loss_box", 0.0))
                agg["obj"]  += float(logs.get("loss_obj", 0.0))
                agg["ang"]  += float(logs.get("loss_ang", 0.0))
                agg["kpt"]  += float(logs.get("loss_kpt", 0.0))
                agg["kc"]   += float(logs.get("loss_kc", 0.0))
                agg["cls"]  += float(logs.get("loss_cls", 0.0))
                agg["pos"]  += float(logs.get("num_pos", 0.0))
                n_steps += 1

                if pbar is not None:
                    # live IoU/Pos indications if provided by criterion
                    iou_dbg = logs.get("mean_iou", None)
                    pos_dbg = logs.get("num_pos", None)
                    pbar.set_postfix_str(
                        f"GPU_mem {_gpu_mem_str(device)}  "
                        f"box={logs.get('loss_box', 0.0):.3f}, obj={logs.get('loss_obj', 0.0):.3f}, "
                        f"ang={logs.get('loss_ang', 0.0):.3f}, kpt={logs.get('loss_kpt', 0.0):.3f}"
                        + (f", IoU={iou_dbg:.3f}" if iou_dbg is not None else "")
                        + (f", Pos={int(pos_dbg)}" if pos_dbg is not None else "")
                    )
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

            # average (reduce if DDP)
            for k in agg:
                tensor_val = torch.tensor(agg[k], device=device, dtype=torch.float32)
                tensor_val = reduce_tensor(tensor_val, op=dist.ReduceOp.SUM) if is_dist() else tensor_val
                agg[k] = float(tensor_val.item()) / self.world_size
            agg_avg = {k: v / max(1, n_steps) for k, v in agg.items()}

            # scheduler step (per-epoch)
            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except TypeError:
                    # some schedulers require val loss; skip gracefully
                    self.scheduler.step(epoch + 1)

            # epoch log
            if self.rank == 0:
                self.logger.info(
                    f"{epoch+1:>4}/{self.epochs:<4}    GPU_mem {_gpu_mem_str(device)}  "
                    f"box_loss {agg_avg['box']:.4f}  obj_loss {agg_avg['obj']:.4f}  "
                    f"ang_loss {agg_avg['ang']:.4f}  kpt_loss {agg_avg['kpt']:.4f}  "
                    f"kc_loss {agg_avg['kc']:.4f}  Pos {agg_avg['pos']:.1f}"
                )

            # ------------------------------------------------------------- #
            # evaluation (interval / warmup / fast vs full / subset)
            # ------------------------------------------------------------- #
            do_eval = True
            if (epoch + 1) <= warmup_noeval:
                do_eval = False
            if eval_interval > 1 and ((epoch + 1) % eval_interval != 0):
                do_eval = False

            metrics = {}
            if do_eval:
                # choose eval mode
                use_fast = fast_mode_default
                if full_every > 0 and ((epoch + 1) % full_every == 0):
                    use_fast = False  # force full OBB periodically

                # temporarily tweak evaluator settings
                old_iou_type = evaluator.iou_type
                old_rot_nms  = evaluator.dec_args.get("rotated_nms", True)
                old_topk     = evaluator.dec_args.get("topk", 10000)
                old_topk_lvl = evaluator.dec_args.get("topk_per_level", None)
                old_thresh   = evaluator.dec_args.get("score_thresh", 0.0)

                if use_fast:
                    evaluator.iou_type = "aabb"
                    evaluator.dec_args["rotated_nms"] = False
                    evaluator.dec_args["topk"] = min(1200, old_topk if old_topk is not None else 1200)
                    evaluator.dec_args["topk_per_level"] = min(400, old_topk_lvl or 400)
                    evaluator.dec_args["score_thresh"] = max(0.20, old_thresh if old_thresh is not None else 0.0)
                else:
                    evaluator.iou_type = "obb"
                    # leave rotated_nms as is (evaluator already falls back if missing)

                # compute subset budget
                max_images = None
                if 0.0 < subset_frac < 1.0:
                    try:
                        n_imgs = len(val_loader.dataset)
                        max_images = max(1, int(math.ceil(n_imgs * subset_frac)))
                    except Exception:
                        max_images = None

                # run eval on rank 0 only (metrics are not reduced across ranks here)
                if self.rank == 0:
                    with torch.inference_mode(), autocast(device_type="cuda", enabled=self.use_amp):
                        metrics = evaluator.evaluate(ddp_module(model), val_loader, device=device, max_images=max_images)

                    # safe metric read
                    tp   = int(metrics.get("tp_count", metrics.get("tp_count@base", 0)))
                    ppi  = float(metrics.get("pred_per_img_avg", 0.0))
                    map50= float(metrics.get("map50", 0.0))
                    map_ = float(metrics.get("map", 0.0))
                    pck  = float(metrics.get("pck@0.05", 0.0))
                    pcka = float(metrics.get("pck_any@0.05", metrics.get("pck-any@0.05", 0.0)))
                    r01  = float(metrics.get("recall@0.1", 0.0))
                    r03  = float(metrics.get("recall@0.3", 0.0))
                    r05  = float(metrics.get("recall@0.5", 0.0))
                    biou = float(metrics.get("best_iou_mean", 0.0))
                    nimg = int(metrics.get("images", 0))
                    mode = "FAST(AABB)" if use_fast else "FULL(OBB)"

                    self.logger.info(
                        f"{'':>17} {'all':<5}  Mode {mode:<10}  Images {nimg:>5}  "
                        f"mAP50 {map50:.6f}  PCK@0.05 {pck:.6f}  PCK_any@0.05 {pcka:.6f}  "
                        f"TPs {tp}  pred/img {ppi:.1f}  "
                        f"R@0.1 {r01:.2f}  R@0.3 {r03:.2f}  R@0.5 {r05:.2f}  bestIoU {biou:.3f}"
                    )

                # restore evaluator
                evaluator.iou_type = old_iou_type
                evaluator.dec_args["rotated_nms"] = old_rot_nms
                evaluator.dec_args["topk"] = old_topk
                evaluator.dec_args["topk_per_level"] = old_topk_lvl
                evaluator.dec_args["score_thresh"] = old_thresh

            # ------------------------------------------------------------- #
            # checkpointing (rank 0)
            # ------------------------------------------------------------- #
            if self.rank == 0:
                # last
                self._save_checkpoint(os.path.join(self.ckpt_dir, "last.pt"), metrics)

                # best (by selection metric)
                sel_val = self._selection_value(metrics)
                is_better = False
                if sel_val is not None:
                    if self.sel_mode == "max":
                        is_better = sel_val > self.best_metric
                    else:
                        is_better = sel_val < self.best_metric
                if is_better:
                    self.best_metric = sel_val
                    self._save_checkpoint(os.path.join(self.ckpt_dir, "best.pt"), metrics)

            barrier()

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    def _selection_value(self, metrics: Dict[str, float]) -> Optional[float]:
        if not metrics:
            return None
        # common aliases
        key = self.sel_metric
        if key in metrics:
            return float(metrics[key])
        # some alternate spellings
        aliases = {
            "map@.50": "map50",
            "map50": "map50",
            "map": "map",
            "pck": "pck@0.05",
            "pck@0.05": "pck@0.05",
        }
        k = aliases.get(key, None)
        return float(metrics[k]) if (k and k in metrics) else None

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
