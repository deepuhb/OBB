
# src/data/smoke_test_data.py
"""
Quick smoke test for the data pipeline.

Usage:
    PYTHONPATH=. python src/data/smoke_test_data.py --root datasets/your_set --img_size 640 --limit 2
"""
from __future__ import annotations
import argparse
from torch.utils.data import DataLoader, Subset
from src.data.datasets import YoloObbKptDataset
from src.data.collate import collate_obbdet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="dataset root containing images/ and labels/")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--limit", type=int, default=2)
    args = ap.parse_args()

    ds = YoloObbKptDataset(root=args.root, split=args.split, img_size=args.img_size)
    if args.limit > 0:
        ds = Subset(ds, list(range(min(args.limit, len(ds)))))

    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_obbdet, num_workers=0)

    for bi, batch in enumerate(dl):
        print(f"--- Batch {bi} ---")
        print("image:", tuple(batch["image"].shape))
        for i in range(len(batch["image"])):
            bboxes = batch["bboxes"][i]
            labels = batch["labels"][i]
            kpts = batch["kpts"][i]
            quads = batch["quads"][i]
            meta = batch.get("meta", [{}])[i] if isinstance(batch.get("meta", None), list) else {}
            print(f" sample[{i}]")
            print("  bboxes:", tuple(bboxes.shape), "labels:", tuple(labels.shape), "kpts:", tuple(kpts.shape), "quads:", tuple(quads.shape))
            paths = batch.get("paths", batch.get("path", ['?']))
            path_i = paths[i] if isinstance(paths, list) else paths
            print("  path:", path_i)
            print("  curr_size:", meta.get("curr_size", None))
        break

    print("OK: data pipeline smoke test passed")


if __name__ == "__main__":
    main()
