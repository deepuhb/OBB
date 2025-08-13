# src/data/build.py
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .yolo_obb_kpt_dataset import YoloObbKptDataset
from ..engine.dist import is_dist

def collate(batch):
    imgs = [b["image"] for b in batch]
    out = {"image": torch.stack(imgs, 0)}
    for k in ("boxes", "quads", "kpts", "angles", "labels", "orig_size"):
        out[k] = [b[k] for b in batch]
    return out

def build_loaders_splits(cfg, rank=0):
    root = cfg.data.root
    img_size = int(cfg.data.img_size)
    workers = int(cfg.data.workers)
    pin = bool(cfg.data.pin_memory)

    batch = int(cfg.train.batch)
    tr_ds = YoloObbKptDataset(root, split="train", img_size=img_size)
    va_ds = YoloObbKptDataset(root, split="val", img_size=img_size)
    if len(tr_ds) == 0:
        tr_ds = YoloObbKptDataset(root, split=None, img_size=img_size)
    if len(va_ds) == 0:
        va_ds = YoloObbKptDataset(root, split=None, img_size=img_size)

    if is_dist():
        tr_samp = DistributedSampler(tr_ds, shuffle=True, drop_last=False)
        va_samp = DistributedSampler(va_ds, shuffle=False, drop_last=False)
        shuffle = False
    else:
        tr_samp = None
        va_samp = None
        shuffle = True

    tr_dl = DataLoader(
        tr_ds, batch_size=batch, shuffle=shuffle, sampler=tr_samp,
        num_workers=workers, pin_memory=pin, collate_fn=collate, drop_last=False
    )
    va_dl = DataLoader(
        va_ds, batch_size=max(1, batch), shuffle=False, sampler=va_samp,
        num_workers=max(0, workers // 2), pin_memory=pin, collate_fn=collate, drop_last=False
    )
    return tr_ds, tr_dl, tr_samp, va_ds, va_dl, va_samp
