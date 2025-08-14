# scripts/train.py
from __future__ import annotations
import argparse
import os
import math
import random
import time
import yaml
from types import SimpleNamespace
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import GradScaler
from torch.backends import cudnn

# ---- project imports (expected to exist in your repo) ----
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator
from src.data.build import build_dataloaders
# Top-down OBB+1KP model and loss
from src.models.yolo11_obbpose_td import YOLO11_OBBPOSE_TD
from src.models.losses.td_obb_kpt1_loss import TDOBBWKpt1Criterion

# --------------------------- utils ---------------------------

def setup_logger(name: str = "obbpose11", level: int = 20):
    import logging
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(fmt="%(asctime)s  %(message)s", datefmt="[%H:%M:%S]"))
    lg.addHandler(h)
    lg.setLevel(level)
    lg.propagate = False
    return lg

def seed_everything(seed: int, rank: int = 0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_cfg(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    # convert nested dicts to SimpleNamespace
    def to_ns(x):
        if isinstance(x, dict):
            return SimpleNamespace(**{k: to_ns(v) for k, v in x.items()})
        elif isinstance(x, list):
            return [to_ns(v) for v in x]
        else:
            return x
    return to_ns(d or {})

def merge_overrides(cfg: SimpleNamespace, args: argparse.Namespace):
    # train overrides
    if not hasattr(cfg, "train"):
        cfg.train = SimpleNamespace()
    if args.batch is not None:
        cfg.train.batch = int(args.batch)
    if args.batch_mode is not None:
        cfg.train.batch_mode = str(args.batch_mode)
    if args.accum is not None:
        cfg.train.accum = int(args.accum)
    if args.epochs is not None:
        cfg.train.epochs = int(args.epochs)
    if args.workers is not None:
        cfg.train.workers = int(args.workers)
    if args.lr is not None:
        cfg.train.lr = float(args.lr)
    if args.weight_decay is not None:
        cfg.train.weight_decay = float(args.weight_decay)
    if args.amp is not None:
        cfg.train.amp = bool(args.amp)
    if args.save_dir is not None:
        cfg.train.save_dir = args.save_dir

    # data overrides
    if not hasattr(cfg, "data"):
        cfg.data = SimpleNamespace()
    if args.overfit_n is not None:
        cfg.data.overfit_n = int(args.overfit_n)

    # eval overrides (subset/interval)
    if not hasattr(cfg, "eval"):
        cfg.eval = SimpleNamespace()
    if args.eval_interval is not None:
        cfg.eval.interval = int(args.eval_interval)
    if args.eval_subset is not None:
        cfg.eval.subset = float(args.eval_subset)

    return cfg

def init_distributed(backend: str = "nccl"):
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=60)
        )

def get_ddp_info():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, rank, world_size

def per_device_batch(cfg_batch: int, world_size: int, mode: str) -> int:
    """Return batch size per process (GPU)."""
    if mode == "total":
        return max(1, math.ceil(cfg_batch / max(1, world_size)))
    # default: 'per_device'
    return int(cfg_batch)

def dataset_summary(logger, train_loader, val_loader, rank: int):
    if rank != 0:
        return
    try:
        tr_n = len(train_loader.dataset)
    except Exception:
        tr_n = "?"
    try:
        va_n = len(val_loader.dataset)
    except Exception:
        va_n = "?"
    logger.info("DATASET SUMMARY (global, not sharded)")
    logger.info(f"Train: images={tr_n}")
    logger.info(f"Val  : images={va_n}")

# --------------------------- main ---------------------------

def parse_args():
    ap = argparse.ArgumentParser("OBB-Pose11 training (DDP-ready)")
    ap.add_argument("--cfg", type=str, default="src/configs/obbpose11.yaml", help="YAML config path")
    ap.add_argument("--batch", type=int, default=None, help="batch size (interpreted by --batch_mode)")
    ap.add_argument("--batch_mode", type=str, default=None, choices=["per_device", "total"],
                    help="per_device: value is per GPU; total: value is global then split across GPUs")
    ap.add_argument("--accum", type=int, default=None, help="grad accumulation steps")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--amp", type=int, default=None, help="1/0 to enable/disable AMP")
    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--overfit_n", type=int, default=None, help="use first N images of train/val (debug/overfit)")
    ap.add_argument("--eval_interval", type=int, default=None, help="run eval every K epochs")
    ap.add_argument("--eval_subset", type=float, default=None, help="fraction (0-1] of val to evaluate each eval")

    # torchrun will pass LOCAL_RANK env; no explicit --local_rank needed
    return ap.parse_args()

def build_model(cfg: SimpleNamespace, device: str):
    # pull top-down knobs with safe defaults
    td = getattr(cfg, "topdown", SimpleNamespace())
    kpt_crop  = int(getattr(td, "crop_size", 64))
    kpt_expand= float(getattr(td, "expand", 1.25))
    kpt_topk  = int(getattr(td, "kpt_topk", 128))
    roi_chunk = int(getattr(td, "roi_chunk", 128))
    score_thr = float(getattr(td, "score_thresh_kpt", 0.25))

    # classes
    nc = None
    if hasattr(cfg, "data") and hasattr(cfg.data, "nc"):
        nc = int(cfg.data.nc)
    elif hasattr(cfg, "model") and hasattr(cfg.model, "nc"):
        nc = int(cfg.model.nc)
    if nc is None:
        nc = 1

    # optional depth/width scaling from cfg.model
    md = getattr(cfg, "model", SimpleNamespace())
    depth_mult = float(getattr(md, "depth_mult", 1.0))
    width_mult = float(getattr(md, "width_mult", 1.0))

    model = YOLO11_OBBPOSE_TD(
        num_classes=nc,
        depth=depth_mult,
        width=width_mult,
        kpt_crop=kpt_crop,
        kpt_expand=kpt_expand,
        kpt_topk=kpt_topk,
        roi_chunk=roi_chunk,
        score_thresh_kpt=score_thr,
    ).to(device)
    return model

def build_criterion(cfg: SimpleNamespace, device: str):
    loss_cfg = getattr(cfg, "loss", SimpleNamespace())
    # typical weights; read from YAML if present
    w_box = float(getattr(loss_cfg, "w_box", 1.0))
    w_obj = float(getattr(loss_cfg, "w_obj", 1.0))
    w_ang = float(getattr(loss_cfg, "w_ang", 0.5))
    w_kpt = float(getattr(loss_cfg, "w_kpt", 1.0))
    w_kc  = float(getattr(loss_cfg, "w_kc", 0.1))  # keypoint "presence"/confidence (if used)
    # angle mode: 'sin-cos' (default) or 'radian' etc.
    ang_mode = str(getattr(loss_cfg, "angle_mode", "sin-cos"))
    criterion = TDOBBWKpt1Criterion(
        w_box=w_box, w_obj=w_obj, w_ang=w_ang, w_kpt=w_kpt, w_kc=w_kc, angle_mode=ang_mode
    ).to(device)
    return criterion

def main():
    args = parse_args()
    logger = setup_logger()

    # init DDP
    local_rank, rank, world = get_ddp_info()
    use_cuda = torch.cuda.is_available()
    device = f"cuda:{local_rank}" if use_cuda else "cpu"
    if use_cuda:
        torch.cuda.set_device(local_rank)
        cudnn.benchmark = True  # variable input sizes => faster

    init_distributed(backend="nccl" if use_cuda else "gloo")

    # config
    cfg = load_cfg(args.cfg)
    cfg = merge_overrides(cfg, args)

    # seed
    seed = int(getattr(getattr(cfg, "train", SimpleNamespace()), "seed", 0))
    seed_everything(seed, rank=rank)

    # data loaders ----------------------------------------------------------
    # Determine per-process batch size
    tr = getattr(cfg, "train", SimpleNamespace())
    batch_mode = str(getattr(tr, "batch_mode", "per_device"))
    batch_cfg  = int(getattr(tr, "batch", 8))
    batch_per_proc = per_device_batch(batch_cfg, world, batch_mode)
    workers = int(getattr(tr, "workers", 4))

    # overfit_n (optionally limit dataset sizes for debugging)
    overfit_n = int(getattr(getattr(cfg, "data", SimpleNamespace()), "overfit_n", 0))

    if rank == 0:
        if batch_mode == "total":
            eff_global_batch = batch_per_proc * max(1, world)
            logger.info(f"[DDP] world={world} | per_gpu_batch={batch_per_proc} "
                        f"| accum_steps={int(getattr(tr,'accum',1))} | effective_global_batch={eff_global_batch}")
        else:
            eff_global_batch = batch_per_proc * max(1, world)
            logger.info(f"[DDP] world={world} | per_gpu_batch={batch_per_proc} "
                        f"| accum_steps={int(getattr(tr,'accum',1))} | effective_global_batch={eff_global_batch}")

    train_loader, val_loader, train_sampler = build_dataloaders(
        cfg=cfg,
        batch_per_device=batch_per_proc,
        workers=workers,
        overfit_n=overfit_n if overfit_n > 0 else None,
        rank=rank,
        world_size=world,
    )

    dataset_summary(logger, train_loader, val_loader, rank)

    # model / criterion -----------------------------------------------------
    model = build_model(cfg, device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank] if use_cuda else None,
                    output_device=local_rank if use_cuda else None,
                    find_unused_parameters=False, gradient_as_bucket_view=True)

    criterion = build_criterion(cfg, device)

    # optimizer / scaler / scheduler ---------------------------------------
    lr = float(getattr(tr, "lr", 1e-3))
    wd = float(getattr(tr, "weight_decay", 5e-2))
    epochs = int(getattr(tr, "epochs", 100))
    accum = int(getattr(tr, "accum", 1))
    use_amp = bool(getattr(tr, "amp", True))
    save_dir = getattr(tr, "save_dir", "runs/train/exp")
    os.makedirs(save_dir, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    steps_per_epoch = max(1, len(train_loader))
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1e4,
    )

    scaler = GradScaler("cuda" if use_cuda else "cpu", enabled=use_amp)

    # evaluator -------------------------------------------------------------
    evaluator = Evaluator(cfg, debug=False)

    # trainer ---------------------------------------------------------------
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        epochs=epochs,
        use_amp=use_amp,
        scheduler=scheduler,
        grad_accum=accum,
        max_grad_norm=getattr(tr, "max_grad_norm", None),
        logger=logger if rank == 0 else None,  # only rank0 logs to console
        cfg=cfg,
        ckpt_dir=save_dir,
    )

    # train -----------------------------------------------------------------
    trainer.fit(train_loader, val_loader, evaluator, train_sampler=train_sampler)

    # cleanup ---------------------------------------------------------------
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
