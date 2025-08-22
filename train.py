#!/usr/bin/env python3
import os
import time
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.build import build_dataloaders
from src.engine.trainer import Trainer
from src.engine.dist import ddp_is_enabled, get_rank, get_world_size, setup_ddp, cleanup_ddp
from src.engine.checkpoint import load_smart_state_dict, save_checkpoint_bundle

from src.models.yolo11_obbpose_td import YOLO11_OBBPOSE_TD
from src.models.losses.td_obb_kpt1_loss import TDOBBWKpt1Criterion


def parse_args():
    p = argparse.ArgumentParser("YOLO11-OBBPOSE TD — train")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--img_size", type=int, default=640)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=16, help="per-GPU batch size")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--classes", type=int, default=1)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--resume", type=str, default="", help="optional checkpoint path")
    p.add_argument("--save_dir", type=str, default="runs")
    p.add_argument("--name", type=str, default="", help="subfolder name (optional)")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--overfit_n", type=int, default=0, help="debug: use N images repeatedly")
    # evaluation cadence
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--warmup_noeval", type=int, default=0)
    p.add_argument("--metric", type=str, default="mAP50")
    p.add_argument("--metric_mode", type=str, default="max", choices=["max", "min"])
    return p.parse_args()


def build_model(num_classes: int, width: float, device: torch.device):
    model = YOLO11_OBBPOSE_TD(num_classes=num_classes, width=width).to(device)
    # Compact but stable(ish) architecture signature for train/eval matching
    try:
        total = sum(p.numel() for p in model.parameters())
        sig = f"{hash(tuple(n for n, _ in model.named_parameters())) & 0xFFFFFFFF:08x}"
        if get_rank() == 0:
            print(f"[SCHEMA TRAIN] sig={sig} params={total}")
    except Exception:
        pass
    return model


def make_criterion(args, device):
    """
    Be resilient to small signature diffs between loss versions.
    Prefer the newer names; fall back to the older ones if needed.
    """
    # Preferred / current naming
    try:
        crit = TDOBBWKpt1Criterion(
            num_classes=args.classes,
            strides=(8, 16, 32),
            # routing thresholds between P3/P4/P5 in pixels (short side of OBB bbox)
            level_boundaries=(64.0, 128.0),
            # detection loss weights
            lambda_box=1.0,
            lambda_obj=1.0,
            lambda_ang=0.25,
            lambda_cls=0.0 if args.classes == 1 else 0.5,
            # keypoint path
            lambda_kpt=1.0,
            kpt_freeze_epochs=0,
            kpt_warmup_epochs=0,
            # positives/assignment behavior
            neighbor_cells=True,
            neighbor_range=1,
            adjacent_level_margin=16.0,
        ).to(device)
        return crit
    except TypeError:
        # Legacy naming
        crit = TDOBBWKpt1Criterion(
            num_classes=args.classes,
            strides=(8, 16, 32),
            assign_thr=(64, 128),
            det_box_weight=1.0,
            det_obj_weight=1.0,
            det_ang_weight=0.25,
            kpt_weight=1.0,
            # legacy kpt sampler knobs (if present in your impl they'll be used)
            kpt_topk=256,
            kpt_crop=3,
            kpt_min_per_img=2,
            kpt_fallback_frac=0.4,
        ).to(device)
        return crit


def main():
    # Minor memory friendliness & determinism niceties
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    args = parse_args()

    # --- DDP setup ---
    setup_ddp()
    rank, world = get_rank(), get_world_size()
    is_ddp = ddp_is_enabled()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # --- run dir ---
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = args.name or f"exp-{ts}"
    save_dir = Path(args.save_dir) / name
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(f"[DDP] world={world} | per_gpu_batch={args.batch} | accum_steps=1")
        print(f"[EVAL cfg] interval={args.eval_interval}  warmup_noeval={args.warmup_noeval}  "
              f"select='{args.metric}' mode='{args.metric_mode}'")

    # --- dataloaders ---
    cfg = {
        "data_root": args.data_root,
        "img_size": args.img_size,
        "classes": args.classes,
        "save_dir": str(save_dir),
        "eval_interval": args.eval_interval,
        "warmup_noeval": args.warmup_noeval,
        "eval_select": args.metric,      # name used inside Trainer
        "eval_mode": args.metric_mode,   # 'max' or 'min'
        "amp": args.amp,
    }
    train_loader, val_loader, train_sampler = build_dataloaders(
        cfg=cfg,
        batch_per_device=args.batch,
        workers=args.workers,
        overfit_n=args.overfit_n,
        rank=rank,
        world_size=world,
    )

    # --- model / loss / opt ---
    model = build_model(args.classes, args.width, device)
    if is_ddp:
        # find_unused_parameters=True: kpt path may be unused for some micro-batches
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    criterion = make_criterion(args, device)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0.1 * args.lr)

    # --- optional resume (load on ALL ranks so params/buffers match) ---
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_file():
            try:
                # Handle common key prefix drift (e.g. 'neck.inner.' -> 'neck.')
                load_smart_state_dict(model.module if is_ddp else model,
                                      ckpt_path, map_renames=[("neck.inner.", "neck.")])
                if rank == 0:
                    print(f"[RESUME] loaded weights from {ckpt_path}")
            except Exception as e:
                if rank == 0:
                    print(f"[RESUME] failed to load weights: {e}")
        elif rank == 0:
            print(f"[RESUME] path not found: {ckpt_path}")

    # --- trainer ---
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=opt,
        scheduler=sched,
        device=device,
        cfg=cfg,
        logger=None,
    )

    # --- fit ---
    best = trainer.fit(train_loader, val_loader, evaluator=None, train_sampler=train_sampler)

    # --- save final bundle on rank-0 ---
    if rank == 0:
        save_checkpoint_bundle(
            save_dir / "last.pt",
            model=model,
            optimizer=opt,
            scheduler=sched,
            meta={
                "epochs": args.epochs,
                "args": vars(args),
                "best": best,
            },
        )

    cleanup_ddp()


if __name__ == "__main__":
    main()
