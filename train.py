
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train YOLO11 OBB+KPT (DFL-only)
- Model: YOLO11_OBBPOSE_TD
- Loss:  TDOBBWKpt1Criterion (DFL-only, single-logit angle)
- Evaluator: Evaluator (expects cfg dict)
This matches the current signatures under src/* as uploaded.
"""

import os
import argparse
import torch
import torch.distributed as dist

from src.models.yolo11_obbpose_td import YOLO11_OBBPOSE_TD
from src.models.losses.td_obb_kpt1_loss import TDOBBWKpt1Criterion
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator
from src.utils.ddp import init_distributed_mode, cleanup_distributed
from src.utils.general import set_seed


def parse_args():
    ap = argparse.ArgumentParser("Train YOLO11 OBB+KPT (DFL-only)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32, help="total batch per GPU")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--classes", type=int, default=1)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--lr0", type=float, default=5e-3)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.937)
    ap.add_argument("--warmup_epochs", type=float, default=3.0)
    # Loss weights aligned with TDOBBWKpt1Criterion
    ap.add_argument("--lambda_obj", type=float, default=1.0)
    ap.add_argument("--lambda_box", type=float, default=5.0)
    ap.add_argument("--lambda_dfl", type=float, default=0.5)
    ap.add_argument("--lambda_ang", type=float, default=0.25)
    ap.add_argument("--lambda_cls", type=float, default=0.5)
    ap.add_argument("--lambda_kpt", type=float, default=1.0)
    ap.add_argument("--neg_obj_ratio", type=float, default=1.0)
    # Eval / decode
    ap.add_argument("--conf_thres", type=float, default=0.25)
    ap.add_argument("--iou_thres", type=float, default=0.50)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--use_nms", action="store_true", default=True)
    ap.add_argument("--save_dir", type=str, default="./runs/exp")
    ap.add_argument("--eval_interval", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    ap.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    ap.add_argument("--dist_url", type=str, default="env://")
    return ap.parse_args()


def build_model(num_classes: int, width: float, device: torch.device):
    model = YOLO11_OBBPOSE_TD(num_classes=num_classes, width=width).to(device)
    return model


def make_criterion(args, device: torch.device):
    crit = TDOBBWKpt1Criterion(
        num_classes=args.classes,
        lambda_obj=args.lambda_obj,
        lambda_box=args.lambda_box,
        lambda_dfl=args.lambda_dfl,
        lambda_ang=args.lambda_ang,
        lambda_cls=args.lambda_cls,
        lambda_kpt=args.lambda_kpt,
        neg_obj_ratio=args.neg_obj_ratio,
        eps=1e-7,
    )
    return crit.to(device)


def make_evaluator(args):
    cfg = dict(
        interval=args.eval_interval,
        warmup_noeval=0,
        select="mAP50",
        mode="max",
        score_thresh=args.conf_thres,
        iou_thresh=args.iou_thres,
        max_det=args.max_det,
        use_nms=args.use_nms,
    )
    evaluator = Evaluator(cfg=cfg, names=None, logger=None, params=None)
    return evaluator


def main():
    args = parse_args()
    set_seed(args.seed)

    init_distributed_mode(args.rank, args.world_size, args.dist_url)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    model = build_model(args.classes, args.width, device)
    criterion = make_criterion(args, device)
    evaluator = make_evaluator(args)

    trainer = Trainer(
        model=model,
        optimizer_cfg=dict(lr0=args.lr0, lrf=args.lrf, momentum=args.momentum, warmup_epochs=args.warmup_epochs),
        save_dir=args.save_dir,
        ddp_world_size=args.world_size,
    )

    # Build dataloaders via your project's helper
    from src.data.build import build_dataloaders

    train_loader, val_loader, train_sampler = build_dataloaders(
        imgsz=args.imgsz,
        batch_size=args.batch,
        workers=args.workers,
        world_size=args.world_size,
        rank=args.rank,
    )

    best = trainer.fit(train_loader, val_loader, evaluator=evaluator, train_sampler=train_sampler, criterion=criterion)
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"[DONE] best {evaluator.select} = {best:.6f}")
    else:
        print(f"[DONE] best {evaluator.select} = {best:.6f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
