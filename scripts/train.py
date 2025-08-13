# scripts/train.py
import os
import argparse
import random
import numpy as np
import torch

from src.configs.config import load_config
from src.engine.dist import init_distributed, is_dist, is_main_process, get_rank
from src.data.build import build_loaders_splits
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator
import torch.distributed as dist

# import model and loss
from src.models.yolo11_obbpose_td import YOLO11_OBBPOSE_TD
from src.models.losses.td_obb_kpt1_loss import TDOBBWKpt1Criterion



def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def cleanup():
    if is_dist():
        dist.destroy_process_group()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="configs/obbpose11.yaml")
    ap.add_argument("--resume", type=str, default="", help="path to checkpoint .pt (best.pt or last.pt)")
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    init_distributed(backend=cfg.ddp.backend if hasattr(cfg, "ddp") else "nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.set_device(local_rank)

    set_seed(int(cfg.train.seed))

    # Dataloaders (with distributed samplers if DDP)
    tr_ds, tr_dl, tr_samp, va_ds, va_dl, va_samp = build_loaders_splits(cfg, rank=get_rank())
    if is_main_process():
        print("DATASET SUMMARY (global, not sharded)")
        print(f"Train: images={len(tr_ds)}")
        print(f"Val  : images={len(va_ds)}")

    # Model + Loss
    model = YOLO11_OBBPOSE_TD(
        num_classes=int(cfg.model.num_classes),
        width=float(cfg.model.width),
        depth=float(cfg.model.depth),
        kpt_crop=int(getattr(cfg.topdown, "crop_size", 64)),
        kpt_expand=float(getattr(cfg.topdown, "expand", 1.25)),
    )
    criterion = TDOBBWKpt1Criterion(
        strides=tuple(cfg.model.strides),
        num_classes=int(cfg.model.num_classes),
        lambda_box=float(cfg.loss.lambda_box),
        lambda_obj=float(cfg.loss.lambda_obj),
        lambda_ang=float(cfg.loss.lambda_ang),
        lambda_cls=1.0,
        lambda_kpt=float(cfg.loss.lambda_kpt),
        kpt_crop=int(getattr(cfg.topdown, "crop_size", 64)),
        kpt_expand=float(getattr(cfg.topdown, "expand", 1.25)),
        kpt_freeze_epochs=int(getattr(cfg.loss, "kpt_freeze_epochs", 0)),
        kpt_warmup_epochs=int(getattr(cfg.loss, "kpt_loss_warmup_epochs", 0)),
        kpt_iou_gate=float(getattr(cfg.loss, "kpt_iou_gate", 0.0)),
    )

    # Evaluator
    evaluator = Evaluator(cfg, debug=False)

    # Trainer
    trainer = Trainer(model, criterion, cfg, device=device)

    if args.resume:
        trainer.resume(args.resume)

    trainer.fit(tr_dl, va_dl, evaluator, train_sampler=tr_samp)

if __name__ == "__main__":
    main()
    cleanup()