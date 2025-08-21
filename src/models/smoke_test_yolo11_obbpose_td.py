import torch

import os, sys
REPO_ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.yolo11_obbpose_td import YOLO11_OBBPOSE_TD

def main():
    model = YOLO11_OBBPOSE_TD(num_classes=1, width=0.50, depth=0.33).eval()
    imgs = torch.randn(2, 3, 768, 768)

    with torch.no_grad():
        out = model(imgs)

    print("FEATS:", [tuple(t.shape) for t in out["feats"]])      # [(B,C3,H/8,W/8), (B,C4,H/16,W/16), (B,C5,H/32,W/32)]
    print("DETS :", [tuple(t.shape) for t in out["det"]])         # [(B,7+nc,H/8,W/8), ...]
    print("KPM  :", [tuple(t.shape) for t in out["kpt_maps"]])    # [(B,3,H/8,W/8), ...]

    # Dummy OBBs per image [cx,cy,w,h,deg]
    H, W = imgs.shape[-2:]
    obb_list = [
        torch.tensor([[W*0.3, H*0.3, 80, 40,  15.0],
                      [W*0.7, H*0.6, 60,120, -30.0]], dtype=torch.float32),
        torch.tensor([[W*0.5, H*0.5, 90, 90,   0.0]], dtype=torch.float32),
    ]

    with torch.no_grad():
        uv, metas = model.kpt_from_obbs(out["feats"], obb_list, topk=16, chunk=8, score_thresh=0.0)
    print("KPT_FROM_OBBS uv:", tuple(uv.shape), " metas:", len(metas), " sample keys:", list(metas[0].keys()) if metas else None)

    preds = model.decode_obb_from_pyramids(out["det"], imgs, conf_thres=0.5, max_det=50, use_nms=False)

    print("DECODE len:", len(preds), " keys0:", list(preds[0].keys()))
    if len(preds) and "obb" in preds[0]:
        print("DECODE[0] obb shape:", tuple(preds[0]["obb"].shape))

if __name__ == "__main__":
    main()
