"""
Entry point for training the simplified YOLO‑style OBB + keypoint detector.

This script demonstrates how to construct the model, criterion, dataset
and data loader, and run a basic training loop.  It is intentionally
minimal and meant as a starting point; you may need to adjust it for
your dataset, augmentation pipeline and evaluation.  The key points are:

  * Instantiate ``YOLO11OBBPOSETD`` with the desired number of classes and
    distributional regression bins (``reg_max``).
  * Instantiate ``TDOBBKpt1DFLCriterion`` with matching ``reg_max``,
    strides and appropriate loss weights.
  * At each iteration, forward the images through the model, pass
    detection and keypoint maps along with the targets into the criterion,
    backpropagate the loss and update the model parameters.

Evaluation is not included here; you can use the ``decode_obb_from_pyramids``
method of the head to convert detection maps into boxes and compute
metrics such as mAP using your preferred library.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.models.yolo11_obbpose_td import YOLO11OBBPOSETD
from src.models.losses.td_obb_kpt1_loss import TDOBBWKpt1Criterion


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO‑style OBB detector")
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset configuration or root directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of object classes')
    parser.add_argument('--reg_max', type=int, default=8, help='DFL maximum bin index')
    parser.add_argument('--device', type=str, default='cuda', help='Device for training')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size (square)')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    return parser.parse_args()


def main() -> None:
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---------------------------------------------------------------------
    #  Dataset
    #
    # Here we expect a dataset class similar to YOLOv5/Ultralytics format that
    # returns a tuple (images, targets).  Each target must be a dict with
    # ``boxes`` (cx,cy,w,h,theta), ``labels`` and optional ``keypoints``.
    # Replace ``YourDatasetClass`` and its arguments with your own.
    # ---------------------------------------------------------------------
    try:
        from src.data.datasets import YoloObbKptDataset
    except ImportError as e:
        raise ImportError(
            "Dataset module not found. Please provide a dataset with the same interface as YoloObbKptDataset."
        ) from e

    # instantiate dataset; adjust parameters such as split, augmentations, etc.
    train_dataset = YoloObbKptDataset(
        root=args.data,
        split='train',
        img_size=args.img_size,
        augment=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )

    # Create model and loss
    model = YOLO11OBBPOSETD(num_classes=args.num_classes, reg_max=args.reg_max).to(device)
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
        neighbor_range=1,
        use_kpt=True,
        lambda_iou=0.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---------------------------------------------------------------------
    #  Training loop
    # ---------------------------------------------------------------------
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_samples = 0
        for images, targets in train_loader:
            # move to device
            images = images.to(device, dtype=torch.float32)
            # adjust targets structure if necessary
            batch_targets = []
            for t in targets:
                # each element of targets is expected to be dict with keys boxes, labels, keypoints
                tgt_dict: Dict[str, torch.Tensor] = {}
                for k in ('boxes', 'labels', 'keypoints'):
                    if k in t and t[k] is not None:
                        tgt_dict[k] = t[k].to(device)
                batch_targets.append(tgt_dict)
            # forward
            det_maps, kpt_maps = model(images)
            loss, loss_dict = criterion(det_maps, kpt_maps, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
        scheduler.step()
        avg_loss = total_loss / max(total_samples, 1)
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}: loss={avg_loss:.4f} lr={lr:.6f}")
        # TODO: evaluation using decode_obb_from_pyramids and computing mAP

    print("Training complete.")


if __name__ == '__main__':
    main()