# src/data/build.py
from __future__ import annotations
import math
import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset

# Your dataset + collate
from src.data.datasets import YoloObbKptDataset


# ------------------------- DDP helpers -------------------------

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


# ------------------------- seeding (workers) -------------------------

def _seed_worker(worker_id: int):
    """Make dataloader workers deterministically seeded."""
    worker_seed = torch.initial_seed() % 2**32
    import random, numpy as np
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# ------------------------- small utils -------------------------

def _maybe_subset(ds: torch.utils.data.Dataset, n: Optional[int]) -> torch.utils.data.Dataset:
    """Return first N samples as a Subset if n is set and ds is larger."""
    if n is None or n <= 0:
        return ds
    n = min(n, len(ds))
    return Subset(ds, list(range(n)))

def _has_split_dirs(root: str, split: str) -> bool:
    """
    Check if a YOLO-style split exists under root:
      images/{split}/..., labels/{split}/...
    This is a *soft* check; the dataset class still validates paths/files.
    """
    img_dir = os.path.join(root, "images", split)
    lab_dir = os.path.join(root, "labels", split)
    return os.path.isdir(img_dir) and os.path.isdir(lab_dir)

def collate(batch):
    imgs = [b["image"] for b in batch]
    out = {"image": torch.stack(imgs, 0)}
    for k in ("boxes", "quads", "kpts", "angles", "labels", "orig_size"):
        out[k] = [b[k] for b in batch]
    return out


def build_dataloaders(
    cfg,
    batch_per_device: int,
    workers: int,
    overfit_n: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
):
    """
    Build train/val dataloaders (DDP-friendly) with optional overfit/debug subsetting.

    Args:
        cfg: SimpleNamespace (loaded YAML)
        batch_per_device: batch size per *process/GPU*
        workers: dataloader workers
        overfit_n: if set, restrict train/val to first N samples (for debugging)
        rank/world_size: DDP info

    Returns:
        train_loader, val_loader, train_sampler
    """
    root = cfg.data.root
    img_size = int(cfg.data.img_size)
    pin = bool(getattr(cfg.data, "pin_memory", True))

    # Determine split availability
    tr_split = "train" if _has_split_dirs(root, "train") else None
    va_split = "val" if _has_split_dirs(root, "val") else None

    # Datasets
    tr_ds = YoloObbKptDataset(root, split=tr_split, img_size=img_size)
    va_ds = YoloObbKptDataset(root, split=va_split, img_size=img_size)

    # Fallbacks if a declared split is empty
    if len(tr_ds) == 0:
        tr_ds = YoloObbKptDataset(root, split=None, img_size=img_size)
    if len(va_ds) == 0:
        # If val absent, use a small slice of train as "val" to keep pipeline valid.
        if len(tr_ds) > 0:
            va_ds = _maybe_subset(tr_ds, min(128, len(tr_ds)))
        else:
            va_ds = YoloObbKptDataset(root, split=None, img_size=img_size)

    # Optional overfit/debug subset (apply *before* creating samplers)
    if overfit_n is not None and overfit_n > 0:
        tr_ds = _maybe_subset(tr_ds, overfit_n)
        va_ds = _maybe_subset(va_ds, max(1, min(overfit_n // 4, len(va_ds))))

    # Samplers
    if world_size > 1 and is_dist():
        tr_samp = DistributedSampler(tr_ds, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=False)
        va_samp = DistributedSampler(va_ds, num_replicas=world_size, rank=rank,
                                     shuffle=False, drop_last=False)
        tr_shuffle = False
    else:
        tr_samp = None
        va_samp = None
        tr_shuffle = True

    persistent = workers > 0
    g = torch.Generator()
    # Use a rank-unique seed so each worker stream is deterministic but disjoint
    g.manual_seed(3407 + rank)

    # Loaders
    train_loader = DataLoader(
        tr_ds,
        batch_size=batch_per_device,
        shuffle=tr_shuffle,
        sampler=tr_samp,
        num_workers=int(workers),
        pin_memory=pin,
        persistent_workers=persistent,
        collate_fn=collate,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    # Keep val batch reasonably large but at least 1
    val_loader = DataLoader(
        va_ds,
        batch_size=max(1, batch_per_device),
        shuffle=False,
        sampler=va_samp,
        num_workers=max(0, int(workers) // 2),
        pin_memory=pin,
        persistent_workers=persistent,
        collate_fn=collate,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    return train_loader, val_loader, tr_samp
