# src/data/datasets.py
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

        boxes = []  # axis-aligned [x1,y1,x2,y2] in pixels (for aug if needed)
        quads = []  # (4,2) polygon in pixels
        kpts = []  # (2,) keypoint in pixels
        labels = []  # (,) class ids
        angles = []  # θ (radians) from quad_to_cxcywh_angle

        with open(lab_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                cls = int(float(parts[0]))
                vals = list(map(float, parts[1:]))

                # Expect: cls + 8 poly + 2 kpt (all normalized)
                if len(vals) < 10:
                    continue
                poly_norm = np.array(vals[0:8], dtype=np.float32).reshape(4, 2)
                kpt_norm = np.array(vals[8:10], dtype=np.float32)

                # CLAMP to [0,1] to avoid negatives/ >1 (seen in your debug)
                poly_norm = np.clip(poly_norm, 0.0, 1.0)
                kpt_norm = np.clip(kpt_norm, 0.0, 1.0)

                # de-normalize to pixels
                pts = np.empty_like(poly_norm, dtype=np.float32)
                pts[:, 0] = poly_norm[:, 0] * w
                pts[:, 1] = poly_norm[:, 1] * h
                kx = float(kpt_norm[0] * w)
                ky = float(kpt_norm[1] * h)

                # robust quad -> obb (cx,cy,w,h,theta_rad)
                t = torch.from_numpy(pts)
                cx, cy, ww, hh, theta = quad_to_cxcywh_angle(t)  # theta already radians
                # axis-aligned AABB for transforms that expect rectangular boxes
                x1a = float(cx - ww / 2.0)
                y1a = float(cy - hh / 2.0)
                x2a = float(cx + ww / 2.0)
                y2a = float(cy + hh / 2.0)

                boxes.append([x1a, y1a, x2a, y2a])
                angles.append(float(theta))
                quads.append(pts)
                kpts.append([kx, ky])
                labels.append(cls)

        if len(boxes) == 0:
            boxes = np.zeros((0, 4), np.float32)
            quads = np.zeros((0, 4, 2), np.float32)
            kpts = np.zeros((0, 2), np.float32)
            labels = np.zeros((0,), np.int64)
            angles = np.zeros((0,), np.float32)
        else:
            boxes = np.asarray(boxes, np.float32)
            quads = np.stack(quads, 0).astype(np.float32)
            kpts = np.asarray(kpts, np.float32)
            labels = np.asarray(labels, np.int64)
            angles = np.asarray(angles, np.float32)

        sample = {
            "image": img,
            "boxes": boxes,  # AABB in pixels
            "quads": quads,  # polygon in pixels
            "kpts": kpts,  # (N,2) in pixels
            "labels": labels,  # (N,)
            "angles": angles,  # radians
            "path": img_path,
        }

        sample = transform_sample(sample, self.img_size)
        q = sample.get("quads", None)
        if q is not None:
            q_t = q if torch.is_tensor(q) else torch.as_tensor(q, dtype=torch.float32)
            if q_t.numel() > 0:
                q_t = q_t.view(-1, 4, 2)
                obb = []
                for i in range(q_t.shape[0]):
                    cx, cy, ww, hh, th = quad_to_cxcywh_angle(q_t[i])  # theta in radians
                    obb.append([float(cx), float(cy), float(ww), float(hh), float(th)])
                bboxes = torch.tensor(obb, dtype=torch.float32)
            else:
                bboxes = torch.zeros((0, 5), dtype=torch.float32)
        else:
            # Fallback: use AABB + angle if quads were dropped by transforms
            bx = sample.get("boxes", None)
            ang = sample.get("angles", None)
            if bx is None or (torch.is_tensor(bx) and bx.numel() == 0) or (not torch.is_tensor(bx) and len(bx) == 0):
                bboxes = torch.zeros((0, 5), dtype=torch.float32)
            else:
                bx_t = bx if torch.is_tensor(bx) else torch.as_tensor(bx, dtype=torch.float32)
                cx = 0.5 * (bx_t[:, 0] + bx_t[:, 2])
                cy = 0.5 * (bx_t[:, 1] + bx_t[:, 3])
                ww = (bx_t[:, 2] - bx_t[:, 0]).clamp_min_(1.0)
                hh = (bx_t[:, 3] - bx_t[:, 1]).clamp_min_(1.0)
                th = torch.zeros_like(cx) if ang is None else torch.as_tensor(ang, dtype=torch.float32).reshape(-1)
                bboxes = torch.stack([cx, cy, ww, hh, th], dim=1)

        sample["bboxes"] = bboxes  # <— KEY: ensure this exists

        # Standardize types so collate keeps them
        if not torch.is_tensor(sample.get("labels", None)):
            sample["labels"] = torch.as_tensor(sample.get("labels", []), dtype=torch.long)
        if not torch.is_tensor(sample.get("kpts", None)):
            sample["kpts"] = torch.as_tensor(sample.get("kpts", []), dtype=torch.float32)

        # Unify naming for downstream
        if "path" in sample and "paths" not in sample:
            sample["paths"] = sample["path"]

        return sample
