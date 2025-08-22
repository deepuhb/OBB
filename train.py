# scripts/train.py
from __future__ import annotations
import os, sys, argparse, random
from types import SimpleNamespace
from datetime import timedelta, datetime

# Ensure repo root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.distributed as dist

# --- Project imports ---
from src.data.build import build_dataloaders
from src.engine.trainer import Trainer
from src.models.yolo11_obbpose_td import YOLO11_OBBPOSE_TD
from src.models.losses.td_obb_kpt1_loss import TDOBBWKpt1Criterion

try:
    from src.engine.evaluator import Evaluator
except Exception:
    Evaluator = None

# ----------------- utils -----------------
def seed_all(seed: int = 42):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def get_dist_env():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, rank, world_size

def build_model(num_classes: int, width: float = 1.0):
    import inspect
    sig = inspect.signature(YOLO11_OBBPOSE_TD.__init__)
    params = set(sig.parameters.keys())
    kwargs = {}
    if "num_classes" in params: kwargs["num_classes"] = num_classes
    elif "nc" in params:        kwargs["nc"] = num_classes
    elif "classes" in params:   kwargs["classes"] = num_classes
    if "width" in params: kwargs["width"] = width
    for k in ("kpt_channels", "kp_channels", "kpm_channels", "n_keypoints"):
        if k in params: kwargs[k] = 3
    return YOLO11_OBBPOSE_TD(**kwargs)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    # Data/IO
    ap.add_argument("--data_root", type=str, default="datasets")
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--save_dir", type=str, default="runs")
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--resume", type=str, default="")
    # Train
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16, help="per-GPU batch size")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-3)
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--grad_clip", type=float, default=None)
    ap.add_argument("--overfit_n", type=int, default=0, help="subset N images (debug)")
    # Eval/AMP
    ap.add_argument("--eval_interval", type=int, default=1)
    ap.add_argument("--warmup_noeval", type=int, default=0)
    ap.add_argument("--amp", action="store_true"); ap.add_argument("--no_amp", dest="amp", action="store_false"); ap.set_defaults(amp=True)
    # Model
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
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=60))

    if rank == 0:
        print(f"[DDP] world={world_size} | per_gpu_batch={args.batch} | accum_steps={args.accum_steps}")

    # --------------- dataloaders ---------------
    data_cfg = SimpleNamespace(
        data=SimpleNamespace(
            root=args.data_root,
            train="train",
            val="val",
            img_size=args.img_size,
            pin_memory=True,
        ),
        train=SimpleNamespace(
            mosaic=True, mosaic_prob=0.5,
            fliplr=0.5, flipud=0.0,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        ),
    )

    train_loader, val_loader, train_sampler = build_dataloaders(
        data_cfg,
        batch_per_device=args.batch,
        workers=args.workers,
        overfit_n=args.overfit_n,
        rank=rank,
        world_size=world_size,
    )

    # --------------- model / ddp ---------------
    model = build_model(num_classes=args.classes, width=args.width).to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=False,
        )

    # --------------- criterion / optim / sched ---------------
    criterion = TDOBBWKpt1Criterion(num_classes=args.classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scheduler.step_on_iter = False

    # --------------- run dir ---------------
    run_name = args.run_name or datetime.now().strftime("exp-%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # --------------- optional resume ---------------
    if args.resume:
        try:
            from src.engine import checkpoint as ckpt
            ckpt.smart_load(
                path=args.resume,
                model=model,
                optimizer=optimizer,
                scaler=None,
                img_size=args.img_size,
                device=device,
                strict=False,
                verbose=(rank == 0),
                extra_maps=[{"neck.inner.": "neck."}, {"head.": "det_head."}],
            )
        except Exception as e:
            if rank == 0:
                print(f"[WARN] smart_load failed ({e}); falling back to torch.load strict=False")
            sd = torch.load(args.resume, map_location=device)
            if "model" in sd:
                sd = sd["model"]
            try:
                (model.module if hasattr(model, "module") else model).load_state_dict(sd, strict=False)
            except Exception as e2:
                if rank == 0:
                    print(f"[ERROR] resume load_state_dict failed: {e2}")

    evaluator = Evaluator() if Evaluator is not None else None

    # --------------- Trainer config ---------------
    cfg = SimpleNamespace(
        train=SimpleNamespace(
            epochs=args.epochs,
            accum_steps=args.accum_steps,
            amp=args.amp,
            grad_clip=args.grad_clip,
            eval_interval=args.eval_interval,
            warmup_noeval=args.warmup_noeval,
            log_interval=50,
            save_dir=save_dir,
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
