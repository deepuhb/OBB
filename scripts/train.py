#!/usr/bin/env python3
import os
import time
import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.build import build_dataloaders
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator
from src.engine.dist import ddp_is_enabled, get_rank, get_world_size, setup_ddp, cleanup_ddp
from src.engine.checkpoint import load_smart_state_dict, save_checkpoint_bundle

from src.models.yolo11_obbpose_td import YOLO11_OBBPOSE_TD
from src.models.losses.td_obb_kpt1_loss import TDOBBWKpt1Criterion


def parse_args():
    p = argparse.ArgumentParser("YOLO11-OBBPOSE TD — train (DFL-only)")

    # data / runtime
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--img_size", type=int, default=640)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=16, help="per-GPU batch size")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", type=str, default="", help="optional checkpoint path")
    p.add_argument("--save_dir", type=str, default="runs")
    p.add_argument("--name", type=str, default="", help="subfolder name (optional)")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--overfit_n", type=int, default=0, help="debug: use N images repeatedly")

    # model
    p.add_argument("--classes", type=int, default=1)
    p.add_argument("--width", type=float, default=1.0)

    # optimizer
    p.add_argument("--optim", type=str, default="sgd", choices=["adamw", "sgd"])
    p.add_argument("--nesterov", action="store_true", help="use Nesterov with SGD")
    p.add_argument("--weight_decay", type=float, default=5e-4)

    # LR / warmup (YOLO-style)
    p.add_argument("--lr0", type=float, default=None,
                   help="initial LR; if omitted, auto-picks 0.005 (SGD) or 0.002 (AdamW)")
    p.add_argument("--lr", type=float, default=None, help="DEPRECATED alias for --lr0")
    p.add_argument("--lrf", type=float, default=0.01, help="final LR multiplier at last epoch (lr_final = lr0*lrf)")
    p.add_argument("--warmup_epochs", type=float, default=3.0, help="warmup length in epochs (can be fractional)")
    p.add_argument("--momentum", type=float, default=0.937, help="SGD momentum")
    p.add_argument("--warmup_momentum", type=float, default=0.80, help="warmup starting momentum (SGD)")
    p.add_argument("--grad_clip_norm", type=float, default=0.0)

    # eval / decoding
    p.add_argument("--conf_thres", type=float, default=0.01)
    p.add_argument("--iou_thres", type=float, default=0.50)
    p.add_argument("--max_det", type=int, default=300)
    p.add_argument("--multi_label", action="store_true")
    p.add_argument("--agnostic", action="store_true")
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--warmup_noeval", type=int, default=0)
    p.add_argument("--metric", type=str, default="mAP50")
    p.add_argument("--metric_mode", type=str, default="max", choices=["max", "min"])
    p.add_argument("--eval_on_rank0_only", action="store_true",
                   help="evaluate full val set only on rank-0 (recommended)")

    return p.parse_args()


def build_model(num_classes: int, width: float, device: torch.device):
    model = YOLO11_OBBPOSE_TD(num_classes=num_classes, width=width).to(device)
    # Minimal schema/signature print (helpful for sanity)
    try:
        total = sum(p.numel() for p in model.parameters())
        sig = f"{hash(tuple(n for n, _ in model.named_parameters())) & 0xFFFFFFFF:08x}"
        if get_rank() == 0:
            print(f"[SCHEMA TRAIN] sig={sig} params={total}")
            # Optional backbone/neck hints printed inside model.__init__ already (if any)
    except Exception:
        pass
    return model


def make_criterion(args, device):
    """
    IMPORTANT: Matches current TDOBBWKpt1Criterion signature (DFL-only).
    Do NOT pass 'strides' or level/range args; the loss reads them from the model head at forward().
    """
    # Lambda settings: keep cls off for single-class to avoid unnecessary BCE
    lambda_cls = 0.0 if int(args.classes) == 1 else 0.5

    crit = TDOBBWKpt1Criterion(
        num_classes=int(args.classes),
        lambda_obj=1.0,
        lambda_box=1.0,
        lambda_dfl=1.0,
        lambda_ang=0.25,
        lambda_cls=lambda_cls,
        lambda_kpt=0.0,          # keep 0.0 unless kpt path is implemented/enabled
        neg_obj_ratio=4.0,
        eps=1e-7,
    ).to(device)
    return crit


def main():
    # Memory & perf niceties
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

    # --- LR/autos ---
    if args.lr0 is not None:
        lr0 = float(args.lr0)
    elif args.lr is not None:  # legacy alias
        lr0 = float(args.lr)
    else:
        lr0 = 0.005 if args.optim.lower() == "sgd" else 0.002

    cfg = {
        "data_root": args.data_root,
        "img_size": args.img_size,
        "classes": args.classes,
        "save_dir": str(save_dir),
        "amp": args.amp,
        "epochs": int(args.epochs),

        # eval & selection
        "eval_interval": int(args.eval_interval),
        "warmup_noeval": int(args.warmup_noeval),
        "select": str(args.metric),
        "mode": str(args.metric_mode),
        "eval_on_rank0_only": bool(args.eval_on_rank0_only or True),

        # lr/warmup
        "lr0": lr0,
        "lrf": float(args.lrf),
        "warmup_epochs": float(args.warmup_epochs),
        "momentum": float(args.momentum),
        "warmup_momentum": float(args.warmup_momentum),
        "grad_clip_norm": float(args.grad_clip_norm),
    }

    if rank == 0:
        print(f"[OPTIM] {args.optim.upper()}  lr0={cfg['lr0']}  lrf={cfg['lrf']}  warmup_epochs={cfg['warmup_epochs']}")

    # --- dataloaders ---
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
        # find_unused_parameters=True keeps us safe if some branches are conditionally unused
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    criterion = make_criterion(args, device)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optim.lower() == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=cfg["lr0"],
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=bool(args.nesterov),
        )
    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg["lr0"],
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
    scheduler = None  # Trainer will build YOLO-style scheduler internally

    # --- resume (load on ALL ranks so params/buffers match) ---
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
        optimizer=optimizer,
        scheduler=scheduler,  # Trainer will overwrite with build_yolo_scheduler
        device=device,
        cfg=cfg,
    )

    # --- evaluator (DFL-only decoder path expects these names) ---
    evaluator = Evaluator(cfg=dict(
        conf_thres=float(getattr(args, "conf_thres", 0.25)),
        iou_thres=float(getattr(args, "iou_thres", 0.45)),
        max_det=int(getattr(args, "max_det", 300)),
        use_nms=True,
        multi_label=bool(getattr(args, "multi_label", False)),
        agnostic=bool(getattr(args, "agnostic", False)),
        # mAP computation range; keep 0.50..0.50 for mAP@50 (like your logs)
        map_iou_st=0.50, map_iou_ed=0.50, map_iou_step=0.05,
    ))

    # --- fit ---
    best = None
    try:
        best = trainer.fit(train_loader, val_loader, evaluator=evaluator, train_sampler=train_sampler)
    finally:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()

    # --- save final bundle on rank-0 ---
    if rank == 0:
        save_checkpoint_bundle(
            save_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            meta={
                "epochs": int(args.epochs),
                "args": vars(args),
                "best": best,
            },
        )

    cleanup_ddp()


if __name__ == "__main__":
    main()
