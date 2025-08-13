# src/data/build.py
from torch.utils.data import DataLoader
import torch
from .yolo_obb_kpt_dataset import YoloObbKptDataset

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def collate(batch):
    imgs = [b["image"] for b in batch]
    out = {"image": torch.stack(imgs, 0)}
    for k in ("boxes","quads","kpts","angles","labels","orig_size"):
        out[k] = [b[k] for b in batch]
    return out

def build_loaders_splits(root, img_size=768, batch_size=4, workers=0):
    """Return (train_ds, train_dl, val_ds, val_dl). Falls back to flat dirs if split dirs are empty."""
    tr_ds = YoloObbKptDataset(root, split="train", img_size=img_size)
    va_ds = YoloObbKptDataset(root, split="val",   img_size=img_size)
    if len(tr_ds) == 0:
        tr_ds = YoloObbKptDataset(root, split=None, img_size=img_size)
    if len(va_ds) == 0:
        va_ds = YoloObbKptDataset(root, split=None, img_size=img_size)

    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=workers, collate_fn=collate, drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=max(1, batch_size), shuffle=False,
                       num_workers=max(0, workers // 2), collate_fn=collate, drop_last=False)
    return tr_ds, tr_dl, va_ds, va_dl
