# scripts/train.py
from __future__ import annotations

import argparse
import math
import os
import random
from types import SimpleNamespace

import torch
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler
import yaml

# --- project imports ---
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator
from src.data.build import build_dataloaders
from src.utils.distrib import (
    init as dist_init,
    is_main_process,
    maybe_barrier,
    cleanup,
    IS_DISTRIBUTED,
    RANK,
    WORLD_SIZE,
    LOCAL_RANK,
)

# Top-down OBB + 1KP model and loss
from src.models.yolo11_obbpose_td import YOLO11_OBBPOSE_TD
from src.models.losses.td_obb_kpt1_loss import TDOBBWKpt1Criterion


# ---------------------------- utils ----------------------------

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
        data = yaml.safe_load(f) or {}

    def to_ns(x):
        if isinstance(x, dict):
            return SimpleNamespace(**{k: to_ns(v) for k, v in x.items()})
        if isinstance(x, list):
            return [to_ns(v) for v in x]
        return x

    return to_ns(data)


def merge_overrides(cfg: SimpleNamespace, args: argparse.Namespace) -> SimpleNamespace:
    # ensure groups
    if not hasattr(cfg, "train"):
        cfg.train = SimpleNamespace()
    if not hasattr(cfg, "data"):
        cfg.data = SimpleNamespace()
    if not hasattr(cfg, "eval"):
        cfg.eval = SimpleNamespace()

    # train overrides
    for k in ("batch", "accum", "epochs", "workers"):
        v = getattr(args, k, None)
        if v is not None:
            setattr(cfg.train, k, type(v)(v))
    if args.batch_mode is not None:
        cfg.train.batch_mode = str(args.batch_mode)
    if args.lr is not None:
        cfg.train.lr = float(args.lr)
    if args.weight_decay is not None:
        cfg.train.weight_decay = float(args.weight_decay)
    if args.amp is not None:
        cfg.train.amp = bool(args.amp)
    if args.save_dir is not None:
        cfg.train.save_dir = str(args.save_dir)

    # data overrides
    if args.overfit_n is not None:
        cfg.data.overfit_n = int(args.overfit_n)

    # eval overrides
    if args.eval_interval is not None:
        cfg.eval.interval = int(args.eval_interval)
    if args.eval_subset is not None:
        cfg.eval.subset = float(args.eval_subset)

    return cfg


def dataset_summary(logger, train_loader, val_loader):
    if not is_main_process():
        return
    def n(x):
        try:
            return len(x.dataset)
        except Exception:
            return "?"
    logger.info(f"DATASET SUMMARY — train images: {n(train_loader)} | val images: {n(val_loader)}")


def per_device_batch(cfg_batch: int, world_size: int, mode: str) -> int:
    """
    Compute per-process batch size.
    mode == "per_device": value used as-is on each process
    mode == "total": value is global and will be split across processes
    """
    if str(mode) == "total":
        return max(1, math.ceil(int(cfg_batch) / max(1, int(world_size))))
    return int(cfg_batch)


def build_model(cfg: SimpleNamespace, device: torch.device):
    td = getattr(cfg, "topdown", SimpleNamespace())
    md = getattr(cfg, "model", SimpleNamespace())

    model = YOLO11_OBBPOSE_TD(
        num_classes=int(getattr(getattr(cfg, "data", SimpleNamespace()), "nc", getattr(md, "nc", 1))),
        depth=float(getattr(md, "depth_mult", 1.0)),
        width=float(getattr(md, "width_mult", 1.0)),
        kpt_crop=int(getattr(td, "crop_size", 64)),
        kpt_expand=float(getattr(td, "expand", 1.25)),
        kpt_topk=int(getattr(td, "kpt_topk", 128)),
        roi_chunk=int(getattr(td, "roi_chunk", 128)),
        score_thresh_kpt=float(getattr(td, "score_thresh_kpt", 0.25)),
    ).to(device)
    return model


def build_criterion(cfg: SimpleNamespace, device: torch.device):
    loss_cfg = getattr(cfg, "loss", SimpleNamespace())
    td = getattr(cfg, "topdown", SimpleNamespace())

    # allow both lambda_* and legacy w_* names
    def pick(obj, *names, default=None, cast=float):
        for n in names:
            if hasattr(obj, n):
                return cast(getattr(obj, n))
        return cast(default) if default is not None else default

    lambda_box = pick(loss_cfg, "lambda_box", "w_box", default=7.5)
    lambda_obj = pick(loss_cfg, "lambda_obj", "w_obj", default=3.0)
    lambda_ang = pick(loss_cfg, "lambda_ang", "w_ang", default=1.0)
    lambda_cls = pick(loss_cfg, "lambda_cls", "w_cls", default=1.0)
    lambda_kpt = pick(loss_cfg, "lambda_kpt", "w_kpt", default=2.0)

    # ROI crop params for the TD keypoint head
    kpt_crop   = int(getattr(td, "crop_size", getattr(loss_cfg, "kpt_crop", 64)))
    kpt_expand = float(getattr(td, "expand", getattr(loss_cfg, "kpt_expand", 1.25)))

    # optional curriculum knobs (the class supports these)
    kpt_freeze_epochs  = int(getattr(loss_cfg, "kpt_freeze_epochs", 0))
    kpt_warmup_epochs  = int(getattr(loss_cfg, "kpt_warmup_epochs", 0))
    kpt_iou_gate       = float(getattr(loss_cfg, "kpt_iou_gate", 0.0))

    criterion = TDOBBWKpt1Criterion(
        num_classes=int(getattr(getattr(cfg, "data", SimpleNamespace()), "nc", 1)),
        lambda_box=lambda_box,
        lambda_obj=lambda_obj,
        lambda_ang=lambda_ang,
        lambda_cls=lambda_cls,
        lambda_kpt=lambda_kpt,
        kpt_crop=kpt_crop,
        kpt_expand=kpt_expand,
        kpt_freeze_epochs=kpt_freeze_epochs,
        kpt_warmup_epochs=kpt_warmup_epochs,
        kpt_iou_gate=kpt_iou_gate,
    ).to(device)
    return criterion



# ---------------------------- main ----------------------------

def parse_args():
    ap = argparse.ArgumentParser("OBB-Pose11 training (DDP-safe)")
    ap.add_argument("--cfg", type=str, default="src/configs/obbpose11.yaml", help="YAML config path")
    ap.add_argument("--batch", type=int, default=None, help="batch size (interpreted by --batch_mode)")
    ap.add_argument("--batch_mode", type=str, default=None, choices=["per_device", "total"],
                    help="per_device: value is per GPU; total: global, split across GPUs")
    ap.add_argument("--accum", type=int, default=None, help="grad accumulation steps")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--amp", type=int, default=None, help="1/0 to enable/disable AMP")
    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--overfit_n", type=int, default=None, help="first N images of train/val (debug)")
    ap.add_argument("--eval_interval", type=int, default=None, help="run eval every K epochs")
    ap.add_argument("--eval_subset", type=float, default=None, help="fraction (0-1] of val to evaluate")
    return ap.parse_args()


def main():
    args = parse_args()
    logger = setup_logger()

    # ---- distributed init (safe for single process) ----
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    _, _, _ = dist_init(backend=backend)  # sets IS_DISTRIBUTED/RANK/WORLD_SIZE/LOCAL_RANK

    # device
    if use_cuda:
        torch.cuda.set_device(LOCAL_RANK if IS_DISTRIBUTED else 0)
        device = torch.device(f"cuda:{LOCAL_RANK}" if IS_DISTRIBUTED else "cuda:0")
        cudnn.benchmark = True  # variable input size => faster
    else:
        device = torch.device("cpu")

    # ---- config & seed ----
    cfg = merge_overrides(load_cfg(args.cfg), args)
    seed = int(getattr(getattr(cfg, "train", SimpleNamespace()), "seed", 0))
    seed_everything(seed, rank=RANK)

    # ---- batch/loader params ----
    tr = getattr(cfg, "train", SimpleNamespace())
    batch_mode = str(getattr(tr, "batch_mode", "per_device"))
    batch_cfg = int(getattr(tr, "batch", 8))
    batch_per_proc = per_device_batch(batch_cfg, WORLD_SIZE, batch_mode)
    workers = int(getattr(tr, "workers", 4))
    overfit_n = int(getattr(getattr(cfg, "data", SimpleNamespace()), "overfit_n", 0)) or None

    if is_main_process():
        eff_global_batch = batch_per_proc * max(1, WORLD_SIZE)
        logger.info(
            f"[world={WORLD_SIZE} rank={RANK}] per_gpu_batch={batch_per_proc} "
            f"| accum={int(getattr(tr,'accum',1))} | effective_global_batch={eff_global_batch}"
        )

    # ---- data ----
    train_loader, val_loader, train_sampler = build_dataloaders(
        cfg=cfg,
        batch_per_device=batch_per_proc,
        workers=workers,
        overfit_n=overfit_n,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    dataset_summary(logger, train_loader, val_loader)

    # ---- model & loss ----
    model = build_model(cfg, device)

    if IS_DISTRIBUTED and use_cuda:
        model = DDP(
            model,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    criterion = build_criterion(cfg, device)

    # ---- optim / sched / amp ----
    lr = float(getattr(tr, "lr", 1e-3))
    wd = float(getattr(tr, "weight_decay", 5e-2))
    epochs = int(getattr(tr, "epochs", 100))
    accum = int(getattr(tr, "accum", 1))
    use_amp = bool(getattr(tr, "amp", True))
    save_dir = str(getattr(tr, "save_dir", "runs/train/exp"))
    os.makedirs(save_dir, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    steps_per_epoch = len(train_loader)  # batches on THIS rank
    total_steps = epochs * max(steps_per_epoch, 0)

    # Build a FACTORY that returns a scheduler when given the optimizer.
    # The Trainer will call this AFTER the first optimizer.step().
    from torch.optim.lr_scheduler import ConstantLR, OneCycleLR
    if total_steps < 2:
        def scheduler_factory(opt):
            return ConstantLR(opt, factor=1.0, total_iters=1)
    else:
        def scheduler_factory(opt):
            last = max(0, int(getattr(opt, "_step_count", 0)) - 1)
            return OneCycleLR(
                opt,
                max_lr=lr,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                anneal_strategy="cos",
                div_factor=10.0,
                final_div_factor=1e4,
            )

    scaler = GradScaler(enabled=use_amp)

    # ---- evaluator & trainer ----
    evaluator = Evaluator(cfg, debug=False)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=str(device),
        epochs=epochs,
        use_amp=use_amp,
        scheduler=scheduler_factory,
        grad_accum=accum,
        max_grad_norm=getattr(tr, "max_grad_norm", None),
        logger=logger if is_main_process() else None,
        cfg=cfg,
        ckpt_dir=save_dir,
    )

    trainer.fit(train_loader, val_loader, evaluator, train_sampler=train_sampler)

    # ---- cleanup ----
    maybe_barrier()
    cleanup()


if __name__ == "__main__":
    main()
