
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import os
from typing import Dict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from src.models.yolo11_obbpose_td import YOLO11OBBPOSETD
from src.models.losses.td_obb_kpt1_loss import TDOBBWKpt1Criterion
from src.engine.evaluator import Evaluator

from src.data.datasets import YoloObbKptDataset
from src.data.mosaic_wrapper import AugmentingDataset
from src.data.collate import collate_obbdet


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train YOLO-style OBB + keypoint detector")
    # data/schedule
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--img_size', type=int, default=640)
    p.add_argument('--lr', type=float, default=1e-3)

    # model
    p.add_argument('--num_classes', type=int, default=1)
    p.add_argument('--reg_max', type=int, default=24)
    p.add_argument('--base_ch', type=int, default=32)

    # loss
    p.add_argument('--neighbor_range', type=int, default=1)
    p.add_argument('--lambda_iou', type=float, default=0.0)
    p.add_argument('--use_kpt', action='store_true')

    # runtime
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--no_augment', action='store_true')
    p.add_argument('--amp', action='store_true')

    # eval controls
    p.add_argument('--eval_interval', type=int, default=1)
    p.add_argument('--eval_conf', type=float, default=0.25)
    p.add_argument('--eval_max_det', type=int, default=100)
    p.add_argument('--log_interval', type=int, default=50)

    # distributed
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--rank', type=int, default=int(os.environ.get('RANK', 0)))
    p.add_argument('--world_size', type=int, default=int(os.environ.get('WORLD_SIZE', 1)))
    p.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    return p.parse_args()


def init_distributed(args: argparse.Namespace) -> torch.device:
    ddp = args.world_size > 1
    if ddp:
        from datetime import timedelta
        backend = 'nccl' if args.device.startswith('cuda') and torch.cuda.is_available() else 'gloo'
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=args.rank, world_size=args.world_size,
                                    timeout=timedelta(minutes=60))
        if args.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
        return torch.device(args.device)
    if args.device.startswith('cuda') and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build_loaders(args: argparse.Namespace, distributed: bool):
    train_base = YoloObbKptDataset(root=args.data, split='train', img_size=args.img_size)
    val_ds = YoloObbKptDataset(root=args.data, split='val', img_size=args.img_size)
    train_ds = train_base if args.no_augment else AugmentingDataset(base=train_base)

    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=args.world_size, rank=args.rank,
                                           shuffle=True, drop_last=False)
        val_sampler = None  # rank-0 evaluates full val set
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=bool(args.workers > 0),
        collate_fn=collate_obbdet,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, min(4, args.batch)),
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=bool(args.workers > 0),
        collate_fn=collate_obbdet,
    )
    return train_loader, val_loader, train_sampler


def main() -> None:
    args = get_args()

    # Console logging
    import logging, sys
    logging.basicConfig(
        level=(logging.INFO if args.rank == 0 else logging.WARN),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    print(f"[INIT] rank={args.rank} local_rank={args.local_rank} world_size={args.world_size}")

    distributed = args.world_size > 1
    device = init_distributed(args)

    if device.type == 'cuda':
        print(f"[INIT] Using device {device}, visible idx {args.local_rank}, name={torch.cuda.get_device_name()}")

    train_loader, val_loader, train_sampler = build_loaders(args, distributed)

    evaluator = Evaluator(params=dict(
        conf_thres=args.eval_conf, iou_thres=0.70, max_det=args.eval_max_det, per_image_debug=3
    ))

    # Model
    model = YOLO11OBBPOSETD(
        num_classes=args.num_classes, reg_max=args.reg_max, base_ch=args.base_ch
    ).to(device)

    # To device & channels-last BEFORE DDP
    if device.type == 'cuda':
        model = model.to(device, memory_format=torch.channels_last)

    # DDP
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(
            model,
            device_ids=[args.local_rank] if device.type == 'cuda' else None,
            output_device=args.local_rank if device.type == 'cuda' else None,
            find_unused_parameters=False,   # faster
            gradient_as_bucket_view=False,  # avoid grad stride mismatch warnings
        )

    # Loss
    criterion = TDOBBWKpt1Criterion(
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        strides=(8, 16, 32),
        lambda_box=5.0,
        lambda_obj=1.0,
        lambda_cls=0.5,
        lambda_ang=0.5,
        lambda_dfl=1.0,
        lambda_kpt=2.0,
        level_boundaries=(32.0, 64.0),
        neighbor_cells=True,
        neighbor_range=args.neighbor_range,
        use_kpt=args.use_kpt,
        lambda_iou=args.lambda_iou,
    ).to(device)

    # Optim / sched
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # First-batch GT debug
    did_train_debug = False

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        total_samples = 0
        it_idx = 0

        for batch in train_loader:
            images = batch['image'].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)

            # one-time GT snapshot (detect degrees vs radians)
            if (not did_train_debug) and args.rank == 0:
                try:
                    gtb = batch.get('bboxes', [])
                    if isinstance(gtb, list) and len(gtb):
                        all_gt = [t.to(device) for t in gtb if (t is not None and t.numel())]
                        if all_gt:
                            cat = torch.cat(all_gt, dim=0)
                            if cat.numel():
                                w, h, ang = cat[:, 2], cat[:, 3], cat[:, 4]
                                qw = torch.quantile(w, torch.tensor([0.5, 0.9, 0.99], device=w.device))
                                qh = torch.quantile(h, torch.tensor([0.5, 0.9, 0.99], device=h.device))
                                amax = float(ang.abs().max().item())
                                print(f"[TRAIN GT DEBUG] n={cat.shape[0]}  "
                                      f"w med/p90/p99={qw[0].item():.1f}/{qw[1].item():.1f}/{qw[2].item():.1f}  "
                                      f"h med/p90/p99={qh[0].item():.1f}/{qh[1].item():.1f}/{qh[2].item():.1f}  "
                                      f"|angle|max(rad)={amax:.2f}  |angle|max(deg)={float(torch.rad2deg(ang).abs().max().item()):.1f}")
                                if amax > 3.5:
                                    print("[WARN] Angles likely in DEGREES in labels; convert to radians in your loader.")
                except Exception as e:
                    print(f"[TRAIN GT DEBUG] failed: {e}")
                did_train_debug = True

            # Build targets
            targets_list = []
            for i in range(len(batch['bboxes'])):
                tgt: Dict[str, torch.Tensor] = {
                    'boxes': batch['bboxes'][i].to(device),
                    'labels': batch['labels'][i].to(device),
                }
                if args.use_kpt and ('kpts' in batch):
                    tgt['keypoints'] = batch['kpts'][i].to(device)
                targets_list.append(tgt)

            optimizer.zero_grad(set_to_none=True)

            amp_dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
                det_maps, kpt_maps = model(images)
                loss, loss_dict = criterion(det_maps, kpt_maps, targets_list)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            it_idx += 1

            # live logs
            if args.rank == 0 and (it_idx % max(1, args.log_interval) == 0):
                parts = " ".join(f"{k}={float(v):.3f}" for k, v in loss_dict.items())
                print(f"[train] epoch={epoch} iter={it_idx:05d} bs={bs} loss={loss.item():.3f} {parts}")

        scheduler.step()

        # Epoch summary + Eval (rank-0)
        if args.rank == 0:
            avg_loss = total_loss / max(total_samples, 1)
            cur_lr = scheduler.get_last_lr()[0]
            metrics = {}
            if (epoch % args.eval_interval) == 0:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                metrics = Evaluator.evaluate(evaluator, model, val_loader, device, max_images=None)
            print(f"Epoch {epoch:3d}: loss={avg_loss:.4f} lr={cur_lr:.6f}  "
                  f"mAP50={metrics.get('mAP50', 0.0):.4f}  mAP={metrics.get('mAP', 0.0):.4f}")

        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()
    if args.rank == 0:
        print("Training complete.")
