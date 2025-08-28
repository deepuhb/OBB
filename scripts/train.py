# scripts/train.py
from __future__ import annotations
import argparse
import os
import time
from typing import Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# ---- only this util module (as you requested) ----
from utils.distrib import init as ddp_init, is_main_process, barrier, cleanup

# ---- project imports (keep these as in your repo) ----
from src.models.yolo11_obbpose_td import YOLO11_OBBPOSE_TD
from src.models.losses.td_obb_kpt1_loss import TDOBBWKpt1Criterion
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator

# If your repo has a function to create loaders, import it here.
# I keep the import very narrow and optional to avoid hard assumptions.
def _import_builders():
    """
    Tries a few common places for your loader builders without assuming utils.general exists.
    Adjust one line below to match your repo if needed.
    """
    # Example: src.data.dataset.build_dataloaders(train_bs, val_bs, workers, ...)
    try:
        from src.data.dataset import build_dataloaders   # <- change here if your path/name differs
        return build_dataloaders
    except Exception as e:
        raise ImportError(
            "Could not import your dataloader builder. Please adjust the import inside "
            "train.py:_import_builders() to the actual location in your repo."
        ) from e


def parse_args():
    p = argparse.ArgumentParser("YOLO11 OBB+Kpt training (DFL-only)")
    # model
    p.add_argument("--classes", type=int, required=True, help="number of detection classes")
    p.add_argument("--width", type=float, default=1.0, help="model width multiplier")
    # data
    p.add_argument("--train", type=str, required=True, help="train set descriptor / path")
    p.add_argument("--val", type=str, required=True, help="val set descriptor / path")
    p.add_argument("--imgsz", type=int, default=768, help="train/eval image size")
    p.add_argument("--workers", type=int, default=4)
    # optimization
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=32, help="per-GPU batch size")
    p.add_argument("--accum", type=int, default=1, help="grad accumulation steps")
    p.add_argument("--lr0", type=float, default=5e-3)
    p.add_argument("--lrf", type=float, default=1e-2)
    p.add_argument("--momentum", type=float, default=0.937)
    p.add_argument("--warmup_epochs", type=float, default=3.0)
    # loss weights (kept in sync with current TDOBBWKpt1Criterion)
    p.add_argument("--lambda_box", type=float, default=7.5)
    p.add_argument("--lambda_obj", type=float, default=3.0)
    p.add_argument("--lambda_ang", type=float, default=1.0)
    p.add_argument("--lambda_cls", type=float, default=1.0)
    p.add_argument("--lambda_kpt", type=float, default=2.0)
    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="runs/exp")
    p.add_argument("--sync_bn", action="store_true", help="convert BN to SyncBN in DDP")
    p.add_argument("--local_rank", type=int, default=-1, help="for DDP (torchrun sets this)")
    return p.parse_args()


def set_seed(seed: int):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)


def build_model(num_classes: int, width: float, device: torch.device) -> YOLO11_OBBPOSE_TD:
    model = YOLO11_OBBPOSE_TD(num_classes=num_classes, width=width).to(device)
    return model


def make_criterion(args, model: YOLO11_OBBPOSE_TD, device: torch.device) -> TDOBBWKpt1Criterion:
    # IMPORTANT: do not pass 'lambda_dfl' here (your current file doesn't accept it)
    # Also pass 'strides' explicitly to match your current loss constructor.
    strides = tuple(int(s) for s in getattr(model.head, "strides", (8, 16, 32)))
    crit = TDOBBWKpt1Criterion(
        num_classes=args.classes,
        strides=strides,
        lambda_box=args.lambda_box,
        lambda_obj=args.lambda_obj,
        lambda_ang=args.lambda_ang,
        lambda_cls=args.lambda_cls,
        lambda_kpt=args.lambda_kpt,
    ).to(device)
    return crit


def make_evaluator(args) -> Evaluator:
    # Your previous crash showed Evaluator.__init__ didn’t accept 'cfg'.
    # So create it with no kwargs (works with the evaluator you sent).
    return Evaluator()


def ddp_wrap(model: torch.nn.Module, args, device: torch.device):
    if dist.is_available() and dist.is_initialized():
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=False,
            static_graph=False,
        )
    return model


def main():
    args = parse_args()

    # --- DDP init (only using utils.distrib) ---
    rank, world_size, device = ddp_init()
    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"[DDP] world={world_size} | per_gpu_batch={args.batch} | accum_steps={args.accum}")
        print(f"[OPTIM] SGD  lr0={args.lr0}  lrf={args.lrf}  warmup_epochs={args.warmup_epochs}")

    set_seed(args.seed + (rank if rank is not None else 0))

    # --- Model ---
    model = build_model(args.classes, args.width, device)
    if is_main_process():
        # these prints match your earlier logs
        tot = sum(p.numel() for p in model.parameters())
        print(f"[SCHEMA TRAIN] sig=explicit-pan v2 (DFL-only) params={tot}")
        if hasattr(model.head, "strides"):
            print(f"[SCHEMA TRAIN] backbone_out=({', '.join(str(c) for c in model.backbone_out)}) neck_ch_out={model.neck_ch_out}" if hasattr(model, 'backbone_out') else "")
    model = ddp_wrap(model, args, device)

    # --- Criterion & Evaluator ---
    criterion = make_criterion(args, model.module if hasattr(model, "module") else model, device)
    evaluator = make_evaluator(args)

    # --- DataLoaders (import builder from your repo) ---
    build_dataloaders = _import_builders()
    train_loader, val_loader, train_sampler = build_dataloaders(
        train=args.train,
        val=args.val,
        imgsz=args.imgsz,
        batch_size=args.batch,
        workers=args.workers,
        world_size=world_size,
        rank=rank,
        pin_memory=True,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=None,                # let Trainer build it from args/params if your Trainer supports that,
        # otherwise replace with torch.optim.SGD([...], lr=args.lr0, momentum=args.momentum, nesterov=True)
        save_dir=args.save_dir,
        epochs=args.epochs,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        warmup_epochs=args.warmup_epochs,
        grad_accum=args.accum,
        device=device,
        rank=rank,
        world_size=world_size,
    )

    best = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        evaluator=evaluator,
        train_sampler=train_sampler,
    )

    if is_main_process():
        print(f"[DONE] best={best}")
    cleanup()


if __name__ == "__main__":
    main()
