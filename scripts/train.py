# scripts/train.py
from __future__ import annotations
import os, sys, argparse, random, inspect
from types import SimpleNamespace

# Ensure repo root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from datetime import timedelta

# --- Project imports ---
from src.engine.trainer import Trainer
from src.data.datasets import YoloObbKptDataset
from src.data.mosaic_wrapper import AugmentingDataset
from src.data.collate import collate_obbdet
from src.models.yolo11_obbpose_td import YOLO11_OBBPOSE_TD
from src.models.losses.td_obb_kpt1_loss import TDOBBWKpt1Criterion

try:
    from src.engine.evaluator import Evaluator
except Exception:
    Evaluator = None


def seed_all(seed: int = 42):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_dist_env():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, rank, world_size


def build_loaders(root, img_size, batch_per_gpu, workers,
                  mosaic=True, mosaic_prob=0.5,
                  fliplr=0.5, flipud=0.0,
                  hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                  mixup_prob=0.1, mixup_alpha=0.2,
                  rank=0, world_size=1):
    """Build train/val loaders. Train uses DistributedSampler when world_size>1; val is plain (rank0 only uses it)."""
    # Base datasets
    tr_ds = YoloObbKptDataset(root=root, split="train", img_size=img_size)
    va_ds = YoloObbKptDataset(root=root, split="val",   img_size=img_size)

    # YOLO11-style wrapper for train
    tr_aug = AugmentingDataset(
        base=tr_ds,
        mosaic=mosaic, mosaic_prob=mosaic_prob,
        fliplr=fliplr, flipud=flipud,
        hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v,
        mixup_prob=mixup_prob, mixup_alpha=mixup_alpha,
    )

    # Samplers
    if world_size > 1:
        tr_sampler = DistributedSampler(tr_aug, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    else:
        tr_sampler = None

    tr_loader = DataLoader(
        tr_aug,
        batch_size=batch_per_gpu,
        shuffle=(tr_sampler is None),
        sampler=tr_sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_obbdet,
    )

    # Validation: keep it simple (rank0 consumes it)
    va_loader = DataLoader(
        va_ds,
        batch_size=max(1, batch_per_gpu // 2),
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_obbdet,
    )
    return tr_loader, va_loader, tr_sampler


def build_model(num_classes: int, width: float = 1.0):
    sig = inspect.signature(YOLO11_OBBPOSE_TD.__init__)
    params = set(sig.parameters.keys())
    kwargs = {}

    if "num_classes" in params: kwargs["num_classes"] = num_classes
    elif "nc" in params:        kwargs["nc"] = num_classes
    elif "classes" in params:   kwargs["classes"] = num_classes

    if "width" in params: kwargs["width"] = width
    elif "w" in params:   kwargs["w"] = width

    for k in ("kpt_channels", "kp_channels", "kpm_channels", "n_keypoints"):
        if k in params:
            kwargs[k] = 3  # project uses 3-channel kpt head

    return YOLO11_OBBPOSE_TD(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="datasets")
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16, help="per-GPU batch size")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-3)
    ap.add_argument("--amp", action="store_true"); ap.add_argument("--no_amp", dest="amp", action="store_false"); ap.set_defaults(amp=True)
    ap.add_argument("--eval_interval", type=int, default=1)
    ap.add_argument("--warmup_noeval", type=int, default=0)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--grad_clip", type=float, default=None)
    ap.add_argument("--classes", type=int, default=1)
    ap.add_argument("--width", type=float, default=1.0)
    args = ap.parse_args()

    seed_all(42)

    # --- DDP init & per-rank device ---
    local_rank, rank, world_size = get_dist_env()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    is_distributed = world_size > 1
    if is_distributed and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(minutes=60),
        )

    if rank == 0:
        print(f"[DDP] world={world_size} | per_gpu_batch={args.batch} | accum_steps={args.accum_steps}")

    # Data
    train_loader, val_loader, train_sampler = build_loaders(
        root=args.data_root,
        img_size=args.img_size,
        batch_per_gpu=args.batch,
        workers=args.workers,
        mosaic=True, mosaic_prob=0.5, fliplr=0.5, flipud=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, mixup_prob=0.10, mixup_alpha=0.20,
        rank=rank, world_size=world_size,
    )

    # Model
    model = build_model(num_classes=args.classes, width=args.width).to(device)

    # DDP wrap
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=False,
        )

    # Criterion / Optim / Sched
    criterion = TDOBBWKpt1Criterion(num_classes=args.classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler.step_on_iter = False

    evaluator = Evaluator() if Evaluator is not None else None

    # Trainer config
    cfg = SimpleNamespace(
        train=SimpleNamespace(
            epochs=args.epochs,
            accum_steps=args.accum_steps,
            amp=args.amp,
            grad_clip=args.grad_clip,
            eval_interval=args.eval_interval,
            warmup_noeval=args.warmup_noeval,
            log_interval=50,
        ),
        eval=SimpleNamespace(select="mAP50", mode="max"),
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        cfg=cfg,
        logger=None,
    )

    trainer.fit(train_loader, val_loader, evaluator, train_sampler=train_sampler)

    # Clean shutdown
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
