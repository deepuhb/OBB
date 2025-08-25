#!/usr/bin/env python3
import os
import time
import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.build import build_dataloaders
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator
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
    p.add_argument("--conf_thres", type=float, default=0.25)
    p.add_argument("--iou_thres", type=float, default=0.5)
    p.add_argument("--max_det", type=int, default=300)
    p.add_argument("--multi_label", action="store_true")
    p.add_argument("--agnostic", action="store_true")
    p.add_argument("--classes", type=int, default=1)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
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
    # Compact architecture signature for quick sanity
    try:
        total = sum(p.numel() for p in model.parameters())
        sig = f"{hash(tuple(n for n, _ in model.named_parameters())) & 0xFFFFFFFF:08x}"
        if get_rank() == 0:
            print(f"[SCHEMA TRAIN] sig={sig} params={total}")
    except Exception:
        pass
    return model


def make_criterion(args, device):
    # Prefer newer names; fall back gracefully if not present.
    try:
        crit = TDOBBWKpt1Criterion(
            num_classes=args.classes,
            strides=(8, 16, 32),
            level_boundaries=(64.0, 128.0),
            lambda_box=1.0,
            lambda_obj=1.0,
            lambda_ang=0.25,
            lambda_cls=0.0 if args.classes == 1 else 0.5,
            lambda_kpt=1.0,
            kpt_freeze_epochs=0,
            kpt_warmup_epochs=0,
            neighbor_cells=True,
            neighbor_range=1,
            adjacent_level_margin=16.0,
        ).to(device)
        return crit
    except TypeError:
        crit = TDOBBWKpt1Criterion(
            num_classes=args.classes,
            strides=(8, 16, 32),
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

    # --- base cfg (kept immutable for Trainer) ---
    cfg = {
        "data_root": args.data_root,
        "img_size": args.img_size,
        "classes": args.classes,
        "save_dir": str(save_dir),
        "eval_interval": args.eval_interval,
        "warmup_noeval": args.warmup_noeval,
        "eval_select": args.metric,
        "eval_mode": args.metric_mode,
        "amp": args.amp,
        "epochs": int(args.epochs),  # trainer will freeze this
    }

    # --- dataloaders ---
    # Pass a COPY to the dataloader builder to avoid any in-place cfg mutation leaking back.
    train_loader, val_loader, train_sampler = build_dataloaders(
        cfg=deepcopy(cfg),
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
        cfg=cfg,   # hand the immutable copy
    )

    # --- fit ---
    # Safe defaults if your argparse doesn’t expose these yet
    conf_thres = getattr(args, "conf_thres", 0.25)
    iou_thres = getattr(args, "iou_thres", 0.45)
    max_det = getattr(args, "max_det", 300)
    multi_label = getattr(args, "multi_label", False)
    agnostic = getattr(args, "agnostic", False)

    # mAP@0.50 only (prints as 'mAP50'); change range/step if you want COCO-style
    evaluator = Evaluator(cfg=dict(conf_thres = conf_thres, iou_thres = iou_thres, max_det = max_det,
                                   use_nms = True, multi_label = multi_label, agnostic = agnostic,
                                   map_iou_st = 0.50, map_iou_ed = 0.50, map_iou_step = 0.05,))

    best = trainer.fit(train_loader, val_loader, evaluator=evaluator, train_sampler=train_sampler)

    # --- save final bundle on rank-0 ---
    if rank == 0:
        save_checkpoint_bundle(
            save_dir / "last.pt",
            model=model,
            optimizer=opt,
            scheduler=sched,
            meta={
                "epochs": int(args.epochs),
                "args": vars(args),
                "best": best,
            },
        )

    cleanup_ddp()


if __name__ == "__main__":
    main()