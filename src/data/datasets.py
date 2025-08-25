
# src/data/datasets.py
import os
import cv2
import math
import logging
import numpy as np
import torch
import torch.utils.data as thdata

from .transforms import transform_sample

LOGGER = logging.getLogger("obbpose11.dataset")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Prefer project util if present; else fallback to cv2.minAreaRect
try:
    from src.utils.box_ops import quad_to_cxcywh_angle as _quad2obb_util  # radians
except Exception:
    _quad2obb_util = None

def _quad_to_obb_radians(quad_xy: np.ndarray):
    """Convert a (4,2) quad (pixels) -> (cx,cy,w,h,theta_rad).
    Uses project util if available; otherwise cv2.minAreaRect. Enforces le90 and
    normalizes theta to [-pi/2, pi/2).
    """
    # 1) Try project util
    if _quad2obb_util is not None:
        try:
            res = _quad2obb_util(torch.tensor(quad_xy, dtype=torch.float32))
            # Accept Tensor, tuple/list, or dict-like
            if isinstance(res, torch.Tensor):
                vals = res.detach().cpu().flatten().tolist()
            elif isinstance(res, (tuple, list)):
                vals = list(res)
            elif isinstance(res, dict):
                # common keys
                for keyset in (('cx','cy','w','h','theta'), ('x','y','w','h','a'), ('cx','cy','w','h','angle')):
                    if all(k in res for k in keyset):
                        vals = [res[keyset[0]], res[keyset[1]], res[keyset[2]], res[keyset[3]], res[keyset[4]]]
                        break
                else:
                    vals = None
            else:
                vals = None

            if vals is not None and len(vals) >= 5:
                cx, cy, w, h, th = map(float, vals[:5])
                # enforce le90 if needed
                import math, numpy as np
                if w < h:
                    w, h = h, w
                    th += math.pi / 2.0
                # normalize angle to [-pi/2, pi/2)
                while th < -math.pi/2:
                    th += math.pi
                while th >= math.pi/2:
                    th -= math.pi
                return float(cx), float(cy), float(max(w,1.0)), float(max(h,1.0)), float(th)
        except Exception:
            # fall through to cv2 route
            pass

    # 2) Fallback: OpenCV minAreaRect -> center (x,y), (w,h), angle(deg in [-90,0))
    rect = cv2.minAreaRect(quad_xy.astype(np.float32))
    (cx, cy), (w, h), angle_deg = rect
    # Enforce le90 convention
    if w < h:
        w, h = h, w
        angle_deg = angle_deg + 90.0
    angle_rad = np.deg2rad(angle_deg)
    # Normalize to [-pi/2, pi/2)
    while angle_rad < -math.pi/2:
        angle_rad += math.pi
    while angle_rad >= math.pi/2:
        angle_rad -= math.pi
    return float(cx), float(cy), float(max(w, 1.0)), float(max(h, 1.0)), float(angle_rad)


class YoloObbKptDataset(thdata.Dataset):
    """
    Label format per line (normalized to W,H):
        cls  x1 y1 x2 y2 x3 y3 x4 y4  kx ky
    Outputs (pixels unless stated):
        image: (3,H,W) float tensor (normalized)
        quads: (N,4,2)
        kpts : (N,2)
        labels: (N,)
        bboxes: (N,5) [cx,cy,w,h,theta_rad], theta in [-pi/2, pi/2)
        angles: (N,) same as bboxes[:,4]
        targets: (N,11) [cls + 8*poly + 2*kpt] normalized to current (W,H) for legacy use
        meta: {'orig_size': (H0,W0), 'curr_size': (Hc,Wc), 'path': str}
        paths: list-compatible path key
    """

    def __init__(self, root, split=None, img_size=768):
        super().__init__()
        img_dir = os.path.join(root, "images", split) if split else os.path.join(root, "images")
        lab_dir = os.path.join(root, "labels", split) if split else os.path.join(root, "labels")
        if not (os.path.isdir(img_dir) and os.path.isdir(lab_dir)):
            img_dir = os.path.join(root, "images")
            lab_dir = os.path.join(root, "labels")
        items = []
        for fn in sorted(os.listdir(img_dir)):
            if fn.lower().endswith(IMG_EXTS):
                stem = os.path.splitext(fn)[0]
                lab = os.path.join(lab_dir, stem + ".txt")
                if os.path.exists(lab):
                    items.append((os.path.join(img_dir, fn), lab))
        if len(items) == 0:
            raise FileNotFoundError(f"No (image,label) pairs found under: {img_dir} / {lab_dir}")
        self.items = items
        self.img_size = int(img_size)

    def __len__(self):
        return len(self.items)

    def _read_image(self, path):
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _f32s(xs):
        return [float(x) for x in xs]

    def _parse_label_file(self, path, w, h):
        """Read YOLO-style quads+kpt (normalized)."""
        quads_px, kpts_px, labels = [], [], []
        with open(path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cls = int(float(parts[0]))
                except Exception:
                    LOGGER.warning(f"[labels] bad class in: {path} -> '{parts[:1]}'")
                    continue
                vals = self._f32s(parts[1:])
                if len(vals) < 10:
                    LOGGER.warning(f"[labels] expected 10 floats after cls, got {len(vals)} in {path}")
                    continue
                poly = np.array(vals[0:8], dtype=np.float32).reshape(4, 2)
                kpt  = np.array(vals[8:10], dtype=np.float32)
                poly = np.clip(poly, 0.0, 1.0)
                kpt  = np.clip(kpt,  0.0, 1.0)
                pts = np.empty_like(poly, dtype=np.float32)
                pts[:, 0] = poly[:, 0] * w
                pts[:, 1] = poly[:, 1] * h
                kx = float(kpt[0] * w); ky = float(kpt[1] * h)
                quads_px.append(pts)
                kpts_px.append([kx, ky])
                labels.append(cls)
        return quads_px, kpts_px, labels

    @staticmethod
    def _quad_to_aabb(quad_xy):
        x1 = float(np.min(quad_xy[:, 0])); y1 = float(np.min(quad_xy[:, 1]))
        x2 = float(np.max(quad_xy[:, 0])); y2 = float(np.max(quad_xy[:, 1]))
        return [x1, y1, x2, y2]

    def __getitem__(self, idx):
        img_path, lab_path = self.items[idx]
        img = self._read_image(img_path)
        H0, W0 = img.shape[:2]

        quads_list, kpts_list, labels_list = self._parse_label_file(lab_path, W0, H0)

        if len(quads_list) == 0:
            boxes = np.zeros((0, 4), np.float32)
            quads = np.zeros((0, 4, 2), np.float32)
            kpts  = np.zeros((0, 2), np.float32)
            labels= np.zeros((0,), np.int64)
        else:
            boxes = np.asarray([self._quad_to_aabb(q) for q in quads_list], dtype=np.float32)
            quads = np.stack(quads_list, axis=0).astype(np.float32)
            kpts  = np.asarray(kpts_list, dtype=np.float32)
            labels= np.asarray(labels_list, dtype=np.int64)

        sample = {
            "image": img,
            "boxes": boxes,   # aabb in pixels (for any aux use)
            "quads": quads,   # (N,4,2) pixels
            "kpts":  kpts,    # (N,2) pixels
            "labels": labels, # (N,)
            "path": img_path,
            "meta": {"orig_size": (H0, W0), "path": img_path},
        }

        # resize/pad/normalize
        sample = transform_sample(sample, self.img_size)

        # ensure tensors
        if not torch.is_tensor(sample["image"]):
            im = sample["image"]
            if isinstance(im, np.ndarray):
                im = torch.from_numpy(im.transpose(2, 0, 1)).float()
            sample["image"] = im
        for k, dt, cast in (
            ("labels", np.int64, torch.long),
            ("quads",  np.float32, torch.float32),
            ("boxes",  np.float32, torch.float32),
            ("kpts",   np.float32, torch.float32),
        ):
            if not torch.is_tensor(sample.get(k, None)):
                arr = sample.get(k, np.zeros((0,), dt))
                sample[k] = torch.as_tensor(arr, dtype=cast)

        # derive OBB (cx,cy,w,h,theta_rad) from quads
        quads_t: torch.Tensor = sample["quads"]
        if quads_t.numel() > 0:
            obb_list = []
            ang_list = []
            q_np = quads_t.detach().cpu().numpy().reshape(-1, 4, 2)
            for i in range(q_np.shape[0]):
                cx, cy, ww, hh, th = _quad_to_obb_radians(q_np[i])
                obb_list.append([cx, cy, ww, hh, th])
                ang_list.append(th)
            bboxes = torch.tensor(obb_list, dtype=torch.float32)
            angles = torch.tensor(ang_list, dtype=torch.float32)
        else:
            bboxes = torch.zeros((0, 5), dtype=torch.float32)
            angles = torch.zeros((0,), dtype=torch.float32)

        sample["bboxes"] = bboxes          # (N,5) pixels, theta in radians
        sample["angles"] = angles          # alias of theta (radians)

        # build normalized Nx11 'targets' for legacy paths
        img_t: torch.Tensor = sample["image"]
        _, Hc, Wc = img_t.shape if img_t.dim() == 3 else (3, self.img_size, self.img_size)
        quads_n = sample["quads"].clone()
        if quads_n.numel():
            quads_n[..., 0] /= float(Wc); quads_n[..., 1] /= float(Hc)
            quads_n.clamp_(0.0, 1.0)
            kpts_n = sample["kpts"].clone()
            if kpts_n.numel():
                kpts_n[:, 0] = (kpts_n[:, 0] / float(Wc)).clamp_(0.0, 1.0)
                kpts_n[:, 1] = (kpts_n[:, 1] / float(Hc)).clamp_(0.0, 1.0)
            else:
                kpts_n = torch.zeros((quads_n.shape[0], 2), dtype=torch.float32)
            cls_f = sample["labels"].to(torch.float32).view(-1, 1)
            flat_poly = quads_n.view(quads_n.shape[0], 8)
            targets = torch.cat([cls_f, flat_poly, kpts_n], dim=1)
        else:
            targets = torch.zeros((0, 11), dtype=torch.float32)
        sample["targets"] = targets

        # unify
        if "path" in sample and "paths" not in sample:
            sample["paths"] = sample["path"]
        sample["meta"]["curr_size"] = (int(Hc), int(Wc))

        return sample
