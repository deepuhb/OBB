# src/data/yolo_obb_kpt_dataset.py
import os, cv2, numpy as np, torch
from .transforms import transform_sample
from ..utils.box_ops import quad_to_cxcywh_angle

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

class YoloObbKptDataset(torch.utils.data.Dataset):
    """
    Reads YOLO-style OBB+KPT annotations:
    line: class x1 y1 x2 y2 x3 y3 x4 y4 kx ky   (normalized)
    """
    def __init__(self, root, split=None, img_size=768):
        # Prefer split-specific dirs if present, otherwise flat dirs
        img_dir = os.path.join(root, "images", split) if split else os.path.join(root, "images")
        lab_dir = os.path.join(root, "labels", split) if split else os.path.join(root, "labels")
        if not os.path.isdir(img_dir) or not os.path.isdir(lab_dir):
            # fallback to flat if split dirs not found
            img_dir = os.path.join(root, "images")
            lab_dir = os.path.join(root, "labels")

        self.items = []
        for fn in sorted(os.listdir(img_dir)):
            if fn.lower().endswith(IMG_EXTS):
                stem = os.path.splitext(fn)[0]
                lab = os.path.join(lab_dir, stem + ".txt")
                if os.path.exists(lab):
                    self.items.append((os.path.join(img_dir, fn), lab))
        self.img_size = img_size

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        img_path, lab_path = self.items[idx]
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes=[]; quads=[]; kpts=[]; labels=[]; angles=[]
        with open(lab_path,"r") as f:
            for line in f:
                if not line.strip(): continue
                parts = line.strip().split()
                cls = int(float(parts[0]))
                vals = list(map(float, parts[1:]))
                x1,y1,x2,y2,x3,y3,x4,y4,kx,ky = vals
                pts = np.array([[x1*w,y1*h],[x2*w,y2*h],[x3*w,y3*h],[x4*w,y4*h]], dtype=np.float32)
                t = torch.from_numpy(pts)
                cx,cy,ww,hh,theta = quad_to_cxcywh_angle(t)
                x1a=float(cx-ww/2); y1a=float(cy-hh/2); x2a=float(cx+ww/2); y2a=float(cy+hh/2)
                boxes.append([x1a,y1a,x2a,y2a])
                angles.append(float(theta))
                quads.append(pts)
                kpts.append([kx*w, ky*h])
                labels.append(cls)

        if len(boxes)==0:
            boxes = np.zeros((0,4), np.float32)
            quads = np.zeros((0,4,2), np.float32)
            kpts  = np.zeros((0,2), np.float32)
            labels= np.zeros((0,), np.int64)
            angles= np.zeros((0,), np.float32)
        else:
            boxes = np.array(boxes, np.float32)
            quads = np.stack(quads, 0).astype(np.float32)
            kpts  = np.array(kpts, np.float32)
            labels= np.array(labels, np.int64)
            angles= np.array(angles, np.float32)

        sample = {"image": img, "boxes": boxes, "quads": quads,
                  "kpts": kpts, "labels": labels, "angles": angles}
        return transform_sample(sample, self.img_size)
