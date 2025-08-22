
# scripts/train.py
from __future__ import annotations
import os, sys, math, time, argparse, random, inspect
from types import SimpleNamespace

# Ensure repo root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader

# --- Project imports ---
from src.engine.trainer import Trainer
from src.data.datasets import YoloObbKptDataset
from src.data.mosaic_wrapper import AugmentingDataset
from src.data.collate import collate_obbdet

from src.models.yolo11_obbpose_td import YOLO11_OBBPOSE_TD
from src.models.losses.td_obb_kpt1_loss import TDOBBWKpt1Criterion

# Evaluator
try:
    from src.engine.evaluator import Evaluator
except Exception:
    Evaluator = None


def seed_all(seed: int = 42):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_loaders(root, img_size=640, batch=16, workers=4, mosaic=True, mosaic_prob=0.5,
                  fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                  mixup_prob=0.1, mixup_alpha=0.2):
    # Base datasets
    tr_ds = YoloObbKptDataset(root=root, split="train", img_size=img_size)
    va_ds = YoloObbKptDataset(root=root, split="val", img_size=img_size)

    # YOLO11-style wrapper for train
    tr_aug = AugmentingDataset(
        base=tr_ds,
        mosaic=mosaic, mosaic_prob=mosaic_prob,
        fliplr=fliplr, flipud=flipud,
        hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v,
        mixup_prob=mixup_prob, mixup_alpha=mixup_alpha,
    )

    tr_loader = DataLoader(
        tr_aug, batch_size=batch, shuffle=True, num_workers=workers,
        pin_memory=True, drop_last=True, collate_fn=collate_obbdet,
    )
    va_loader = DataLoader(
        va_ds, batch_size=max(1, batch // 2), shuffle=False, num_workers=workers,
        pin_memory=True, drop_last=False, collate_fn=collate_obbdet,
    )
    return tr_loader, va_loader


def build_model(num_classes: int, width: float = 1.0):
    # Build kwargs only for params supported by the current model signature
    sig = inspect.signature(YOLO11_OBBPOSE_TD.__init__)
    params = set(sig.parameters.keys())  # includes 'self'
    kwargs = {}

    # class count (various aliases across versions)
    if "num_classes" in params:
        kwargs["num_classes"] = num_classes
    elif "nc" in params:
        kwargs["nc"] = num_classes
    elif "classes" in params:
        kwargs["classes"] = num_classes

    # width / scale
    if "width" in params:
        kwargs["width"] = width
    elif "w" in params:
        kwargs["w"] = width

    # optional keypoint map channels (we try several common names)
    for k in ("kpt_channels", "kp_channels", "kpm_channels", "n_keypoints"):
        if k in params:
            kwargs[k] = 3  # your project uses a 3-channel kpt head

    return YOLO11_OBBPOSE_TD(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="datasets", help="dataset root containing images/ and labels/")
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-3)
    ap.add_argument("--amp", action="store_true", help="enable mixed precision (fp16 autocast)")
    ap.add_argument("--no_amp", dest="amp", action="store_false")
    ap.set_defaults(amp=True)
    ap.add_argument("--eval_interval", type=int, default=1)
    ap.add_argument("--warmup_noeval", type=int, default=0)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--grad_clip", type=float, default=None)
    ap.add_argument("--classes", type=int, default=1, help="number of classes")
    args = ap.parse_args()

    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader = build_loaders(
        root=args.data_root, img_size=args.img_size, batch=args.batch, workers=args.workers,
        mosaic=True, mosaic_prob=0.5, fliplr=0.5, flipud=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, mixup_prob=0.10, mixup_alpha=0.20,
    )

    # Model (robust to signature differences)
    model = build_model(num_classes=args.classes, width=1.0).to(device)

    # Criterion
    criterion = TDOBBWKpt1Criterion(num_classes=args.classes)

    # Optimizer & LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine schedule per-epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler.step_on_iter = False  # our Trainer checks this flag

    # Evaluator
    evaluator = Evaluator() if Evaluator is not None else None

    # Trainer config (mirrors the new Trainer API)
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
        eval=SimpleNamespace(
            select="mAP50",
            mode="max",
        ),
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

    trainer.fit(train_loader, val_loader, evaluator, train_sampler=None)


if __name__ == "__main__":
    main()
