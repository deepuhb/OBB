# scripts/train.py
import argparse
import torch
from src.data.build import build_loaders_splits
from src.models.obbpose_model import OBBPoseModel
from src.models.losses.obb_kpt1_loss import OBBKpt1Criterion
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="datasets")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--img_size", type=int, default=768)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    tr_ds, tr_dl, va_ds, va_dl = build_loaders_splits(
        args.data_root, img_size=args.img_size, batch_size=args.batch, workers=args.workers
    )
    print(f"DATA: train={len(tr_ds)}  val={len(va_ds)}")

    model = OBBPoseModel(num_classes=1, width=0.5, depth=0.33)
    criterion = OBBKpt1Criterion(strides=(4,8,16), num_classes=1, assign_4_neighbors=True)
    evaluator = Evaluator(cfg={"eval": {"score_thresh": 0.10, "nms_iou": 0.5, "iou_thr": 0.5, "pck_tau": 0.05},
                               "model": {"strides": (4,8,16)}}, debug=True)

    trainer = Trainer(model, criterion, device=args.device, lr=1e-3, weight_decay=5e-4,
                      use_amp=(args.device.startswith("cuda")), cfg=None)
    trainer.fit(tr_dl, va_dl, evaluator, epochs=args.epochs, soft_warmup=2)

if __name__ == "__main__":
    main()
