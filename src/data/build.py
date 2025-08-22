
# src/data/build.py
from __future__ import annotations
from types import SimpleNamespace
from typing import Tuple, Optional

from torch.utils.data import DataLoader, DistributedSampler, Subset

from src.data.datasets import YoloObbKptDataset          # your dataset
from src.data.mosaic_wrapper import AugmentingDataset    # augmentation wrapper
from src.data.collate import collate_obbdet              # robust collate


def _yaml(cfg, path: str, default=None):
    """Nested getattr with dot-paths, e.g. _yaml(cfg, 'data.img_size', 768)."""
    cur = cfg
    for key in path.split("."):
        if not hasattr(cur, key):
            return default
        cur = getattr(cur, key)
    return cur


def _yaml_names(cfg) -> list[str]:
    n = _yaml(cfg, "data.names", []) or []
    return list(n)


def _maybe_subset(ds, n: Optional[int]):
    if not n or n <= 0:
        return ds
    n = min(n, len(ds))
    return Subset(ds, list(range(n)))


def build_dataloaders(
    cfg: SimpleNamespace,
    batch_per_device: int,
    workers: int,
    overfit_n: Optional[int],
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
    """
    Build train/val dataloaders. Compatible with datasets that only accept (root, split, img_size).
    We DO NOT pass unsupported kwargs like overfit_n/mode/pin_memory.
    """
    # ---- config bits ----
    root = _yaml(cfg, "data.root", "datasets")
    train_split = _yaml(cfg, "data.train", "train")
    val_split = _yaml(cfg, "data.val", "val")
    img_size = int(_yaml(cfg, "data.img_size", 768))
    pin_memory = bool(_yaml(cfg, "data.pin_memory", True))
    yaml_names = _yaml_names(cfg)
    yaml_nc = len(yaml_names) if yaml_names else int(_yaml(cfg, "data.nc", 1))

    # ---- construct base datasets with only safe args ----
    train_base = YoloObbKptDataset(root=root, split=train_split, img_size=img_size)
    val_ds     = YoloObbKptDataset(root=root, split=val_split,   img_size=img_size)

    # Optional small subset for quick debugging
    if overfit_n:
        train_base = _maybe_subset(train_base, overfit_n)
        val_ds     = _maybe_subset(val_ds, overfit_n)

    # ---- push names/nc onto datasets AFTER construction (if provided) ----
    if yaml_names:
        try:
            train_base.names = list(yaml_names)
            val_ds.names = list(yaml_names)
        except Exception:
            pass
    try:
        # If Subset, attach to its dataset
        def _set_nc(obj, nc):
            try:
                obj.num_classes = nc
            except Exception:
                base = getattr(obj, "dataset", None)
                if base is not None:
                    try:
                        base.nc = nc
                    except Exception:
                        pass
        _set_nc(train_base, yaml_nc)
        _set_nc(val_ds, yaml_nc)
    except Exception:
        pass

    # ---- wrap train dataset with augmentations (mosaic + flips + HSV) ----
    trn = getattr(cfg, "train", SimpleNamespace())
    train_ds = AugmentingDataset(
        base=train_base,
        mosaic=bool(getattr(trn, "mosaic", True)),
        mosaic_prob=float(getattr(trn, "mosaic_prob", 0.5)),
        fliplr=float(getattr(trn, "fliplr", 0.5)),
        flipud=float(getattr(trn, "flipud", 0.0)),
        hsv_h=float(getattr(trn, "hsv_h", 0.015)),
        hsv_s=float(getattr(trn, "hsv_s", 0.7)),
        hsv_v=float(getattr(trn, "hsv_v", 0.4)),
    )

    # ---- optional DistributedSampler ----
    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler   = None

    # ---- dataloaders ----
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_per_device,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(workers),
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=bool(workers > 0),
        collate_fn=collate_obbdet,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_per_device,
        shuffle=False,
        sampler=val_sampler,
        num_workers=int(workers),
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=bool(workers > 0),
        collate_fn=collate_obbdet,
    )

    return train_loader, val_loader, train_sampler
