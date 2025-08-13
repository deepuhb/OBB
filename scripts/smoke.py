# scripts/smoke.py
import os, torch, sys
sys.path.insert(0, ".")
from src.data.build import build_loaders
from src.models.obbpose_model import OBBPoseModel
from src.models.losses.obb_kpt1_loss import OBBKpt1Criterion

def main():
    ds, dl = build_loaders("datasets", img_size=768, batch_size=1, workers=0)
    print(f"Dataset size: {len(ds)}")
    batch = next(iter(dl))
    device = "cpu"
    imgs = batch["image"].to(device)
    model = OBBPoseModel(num_classes=1, width=0.5, depth=0.33).to(device)
    crit  = OBBKpt1Criterion(strides=(4,8,16), num_classes=1, assign_4_neighbors=True)
    with torch.no_grad():
        outs = model(imgs)
        det_maps, kpt_maps = outs["det"], outs["kpt"]
        b2 = {k: ([t.to(device) for t in v] if isinstance(v,list) else v.to(device))
              for k,v in batch.items() if k!="image"}
        loss, logs = crit(det_maps, kpt_maps, {"image": imgs, **b2}, epoch=0, soft_warmup_epochs=2)
    print("Single-step loss:", float(loss))
    print("Logs:", logs)
    print("Det map shapes:", [tuple(m.shape) for m in det_maps])
    print("Kpt map shapes:", [tuple(m.shape) for m in kpt_maps])

if __name__ == "__main__":
    main()
