"""
Entry point for training the YOLO‑style oriented bounding box (OBB) + keypoint detector.

This script builds a lightweight YOLO‑like model, loss criterion and dataloaders
for OBB detection with an optional single keypoint.  It supports training on
CPU, a single GPU or multiple GPUs via PyTorch's distributed run utility.  The
training loop can run with or without mosaic/HSV augmentations and supports
automatic mixed precision (AMP) to accelerate training on modern GPUs.

Highlights:

  * Instantiates ``YOLO11OBBPOSETD`` with a configurable number of classes,
    distributional regression bins (``reg_max``) and backbone width (``base_ch``).
  * Instantiates ``TDOBBWKpt1Criterion`` with matching ``reg_max`` and
    assignment parameters such as ``neighbor_range`` and ``lambda_iou``.
  * Optionally wraps the base dataset with a mosaic + mixup augmentation
    wrapper for improved robustness.
  * Supports distributed training via ``torchrun --nproc_per_node=N`` and
    automatic mixed precision via the ``--amp`` flag.

Example usage:

.. code-block:: bash

    # single‑GPU training with augmentations and AMP
    python train.py --data datasets --epochs 100 --batch 16 --img_size 640 \
                   --num_classes 1 --amp

    # distributed training across 4 GPUs
    torchrun --standalone --nproc_per_node=4 train.py \
             --data datasets --epochs 100 --batch 8 --img_size 640 \
             --num_classes 1 --amp

The dataset directory is expected to follow the Ultralytics/YoloV5 format
with subdirectories ``images/{train,val}`` and ``labels/{train,val}``.
Labels must be stored in text files containing one object per line in the
format ``cls x1 y1 x2 y2 x3 y3 x4 y4 kx ky`` where coordinates are
normalized to [0,1].
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from src.models.yolo11_obbpose_td import YOLO11OBBPOSETD
from src.models.losses.td_obb_kpt1_loss import TDOBBWKpt1Criterion

# Data loading utilities; these modules must be present in the repository.
try:
    from src.data.datasets import YoloObbKptDataset
    from src.data.mosaic_wrapper import AugmentingDataset
    from src.data.collate import collate_obbdet
except Exception as e:
    # Degrade gracefully if data modules are missing; the user must supply them.
    YoloObbKptDataset = None  # type: ignore
    AugmentingDataset = None  # type: ignore
    collate_obbdet = None  # type: ignore
    raise ImportError(
        "Data modules not found. Please ensure 'src/data' contains datasets.py, mosaic_wrapper.py and collate.py"
    ) from e


def get_args() -> argparse.Namespace:
    """Parse command‑line arguments for training.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train YOLO‑style OBB + keypoint detector with optional augmentation"
    )
    # Dataset and training schedule
    parser.add_argument('--data', type=str, required=True,
                        help='Root directory of the dataset or YAML config file')
    parser.add_argument('--epochs', type=int, default=100, help='Total number of training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size per device')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of object classes')
    parser.add_argument('--reg_max', type=int, default=8, help='DFL maximum bin index (nbins = reg_max + 1)')
    parser.add_argument('--img_size', type=int, default=640, help='Square input image size after resizing/padding')
    parser.add_argument('--workers', type=int, default=4, help='Number of DataLoader workers per process')
    # Model hyperparameters
    parser.add_argument('--base_ch', type=int, default=32, help='Base channel width for the backbone (controls model size)')
    # Loss/assignment parameters
    parser.add_argument('--neighbor_range', type=int, default=1,
                        help='Manhattan radius around centre cell for positive samples')
    parser.add_argument('--lambda_iou', type=float, default=0.0,
                        help='Weight for IoU penalty term in box regression (0 disables)')
    parser.add_argument('--use_kpt', action='store_true', help='Enable keypoint supervision')
    # Augmentation and precision
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable mosaic/mixup/HSV augmentations')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    # Distributed training parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to train on: 'cuda' or 'cpu'")
    parser.add_argument('--rank', type=int, default=int(os.environ.get('RANK', 0)),
                        help='Global rank for distributed training')
    parser.add_argument('--world_size', type=int, default=int(os.environ.get('WORLD_SIZE', 1)),
                        help='World size for distributed training')
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)),
                        help='Local rank for distributed training')
    return parser.parse_args()


def init_distributed(args: argparse.Namespace) -> torch.device:
    """Initialise distributed training environment if world_size > 1.

    Sets the CUDA device based on ``local_rank`` and initialises the process
    group.  If running on CPU or with a single process, this function is
    effectively a no‑op.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed command‑line arguments.

    Returns
    -------
    torch.device
        The device on which to run the model.
    """
    distributed = args.world_size > 1
    if distributed:
        # Select backend based on available hardware
        backend = 'nccl' if args.device.startswith('cuda') and torch.cuda.is_available() else 'gloo'
        if not dist.is_initialized():
            dist.init_process_group(backend=backend,
                                    rank=args.rank,
                                    world_size=args.world_size)
        # Set the device for this process
        if args.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
        return torch.device(args.device)
    # Non‑distributed: choose CPU or first CUDA device
    if args.device.startswith('cuda') and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build_loaders(args: argparse.Namespace, distributed: bool):
    """Construct training and validation dataloaders.

    Uses the base ``YoloObbKptDataset`` for validation, and optionally wraps
    the training dataset with ``AugmentingDataset`` for mosaic and other
    augmentations.  When distributed training is enabled, returns
    ``DistributedSampler`` objects for both sets.

    Parameters
    ----------
    args: argparse.Namespace
        Training arguments including dataset root, batch size and image size.
    distributed: bool
        Flag indicating whether distributed training is active.

    Returns
    -------
    tuple
        ``(train_loader, val_loader, train_sampler)`` where the sampler is
        either a ``DistributedSampler`` or ``None``.
    """
    # Build base datasets
    train_base = YoloObbKptDataset(root=args.data, split='train', img_size=args.img_size)
    val_ds = YoloObbKptDataset(root=args.data, split='val', img_size=args.img_size)
    # Apply mosaic/mixup/HSV augmentations unless disabled
    if args.no_augment:
        train_ds = train_base
    else:
        train_ds = AugmentingDataset(base=train_base)
    # Distributed sampling
    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None
    # DataLoader options
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
        batch_size=args.batch,
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
    # Initialise distributed environment and select device
    distributed = args.world_size > 1
    device = init_distributed(args)
    # Build dataloaders; returns train_sampler for epoch shuffling
    train_loader, val_loader, train_sampler = build_loaders(args, distributed)
    # Create model
    model = YOLO11OBBPOSETD(num_classes=args.num_classes, reg_max=args.reg_max, base_ch=args.base_ch).to(device)
    # Wrap in DistributedDataParallel if needed
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank] if device.type == 'cuda' else None,
            output_device=args.local_rank if device.type == 'cuda' else None,
        )
    # Create loss criterion
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
    # Optimiser and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Shuffle data each epoch when using a distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_samples = 0
        # Iterate over batches
        for batch in train_loader:
            images = batch['image'].to(device, dtype=torch.float32)
            # Prepare targets list for criterion; convert per‑item tensors
            targets = []
            for i in range(len(batch['bboxes'])):
                tgt: Dict[str, torch.Tensor] = {}
                # oriented boxes: (N,5)
                tgt['boxes'] = batch['bboxes'][i].to(device)
                # class labels: (N,)
                tgt['labels'] = batch['labels'][i].to(device)
                # optional keypoints
                if args.use_kpt and ('kpts' in batch):
                    tgt['keypoints'] = batch['kpts'][i].to(device)
                targets.append(tgt)
            optimizer.zero_grad()
            # Forward + loss under AMP context
            with torch.cuda.amp.autocast(enabled=args.amp):
                det_maps, kpt_maps = model(images)
                loss, loss_dict = criterion(det_maps, kpt_maps, targets)
            # Backward + optimisation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
        # Step learning rate
        scheduler.step()
        # Logging (only by rank 0 to avoid clutter)
        if args.rank == 0:
            avg_loss = total_loss / max(total_samples, 1)
            cur_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d}: loss={avg_loss:.4f} lr={cur_lr:.6f}")
        # Synchronise before next epoch if distributed
        if distributed:
            dist.barrier()
    # Finalise
    if distributed:
        dist.destroy_process_group()
    if args.rank == 0:
        print("Training complete.")


if __name__ == '__main__':
    main()