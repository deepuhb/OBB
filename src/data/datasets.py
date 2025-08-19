# src/data/datasets.py
import os
import cv2
import math
import logging
import numpy as np
import torch
import torch.utils.data as thdata

# project-local imports
from .transforms import transform_sample
from ..utils.box_ops import quad_to_cxcywh_angle

LOGGER = logging.getLogger("obbpose11.dataset")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


class YoloObbKptDataset(thdata.Dataset):
    """
    YOLO-style OBB (+1 keypoint) dataset.

    Per-line label format (normalized to image W,H):
        cls  x1 y1 x2 y2 x3 y3 x4 y4  kx ky
    where (x1,y1) ... (x4,y4) describe the oriented bbox polygon
    (clockwise or CCW), and (kx,ky) is a single keypoint.

    __getitem__ returns a dict with keys:
        image   : FloatTensor (3,H,W)
        bboxes  : (N,5)  [cx,cy,w,h,theta_rad] in *image pixels*
        labels  : (N,)   LongTensor
        kpts    : (N,2)  FloatTensor in *image pixels*
        quads   : (N,4,2) FloatTensor in *image pixels*
        boxes   : (N,4)  FloatTensor axis-aligned [x1,y1,x2,y2] (pixels)
        angles  : (N,)   FloatTensor theta (radians)
        targets : (N,11) FloatTensor [cls + 8*poly + 2*kpt], *normalized* to current W,H
        paths   : str
        meta    : dict (orig_size, curr_size, path)
    """

    def __init__(self, root, split=None, img_size=768):
        super().__init__()
        # Prefer split subdirs if they exist
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
    def _safe_float_list(xs):
        return [float(x) for x in xs]

    def _parse_label_file(self, path, w, h):
        """
        Returns lists for one image:
            quads_px: list of (4,2) np.float32 in *pixels*
            kpts_px : list of (2,)  np.float32 in *pixels*
            labels  : list of int
        All inputs in file are normalized; we clamp to [0,1] to kill negatives/overflow.
        """
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
                    # Skip malformed lines
                    LOGGER.warning(f"[labels] bad class in: {path} -> '{parts[:1]}'")
                    continue

                vals = self._safe_float_list(parts[1:])
                if len(vals) < 10:
                    LOGGER.warning(f"[labels] expected 10 floats, got {len(vals)} in {path}")
                    continue

                # polygon + kpt (normalized)
                poly = np.array(vals[0:8], dtype=np.float32).reshape(4, 2)
                kpt = np.array(vals[8:10], dtype=np.float32)

                # clamp normalized inputs to [0,1]
                poly = np.clip(poly, 0.0, 1.0)
                kpt = np.clip(kpt, 0.0, 1.0)

                # to pixels
                pts = np.empty_like(poly, dtype=np.float32)
                pts[:, 0] = poly[:, 0] * w
                pts[:, 1] = poly[:, 1] * h
                kx = float(kpt[0] * w)
                ky = float(kpt[1] * h)

                quads_px.append(pts)          # (4,2)
                kpts_px.append([kx, ky])      # (2,)
                labels.append(cls)

        return quads_px, kpts_px, labels

    @staticmethod
    def _quad_to_aabb(quad_xy):
        """Axis-aligned [x1,y1,x2,y2] from (4,2) polygon."""
        x1 = float(np.min(quad_xy[:, 0]))
        y1 = float(np.min(quad_xy[:, 1]))
        x2 = float(np.max(quad_xy[:, 0]))
        y2 = float(np.max(quad_xy[:, 1]))
        return [x1, y1, x2, y2]

    def __getitem__(self, idx):
        img_path, lab_path = self.items[idx]

        # -- read image
        img = self._read_image(img_path)
        H0, W0 = img.shape[:2]

        # -- parse labels (into pixel space, pre-transform)
        quads_list, kpts_list, labels_list = self._parse_label_file(lab_path, W0, H0)

        if len(quads_list) == 0:
            # Empty (rare) — create empty structures
            boxes = np.zeros((0, 4), np.float32)
            quads = np.zeros((0, 4, 2), np.float32)
            kpts = np.zeros((0, 2), np.float32)
            labels = np.zeros((0,), np.int64)
            angles = np.zeros((0,), np.float32)
        else:
            # AABB for generic transforms; angles filled after OBB conversion
            boxes = np.asarray([self._quad_to_aabb(q) for q in quads_list], dtype=np.float32)
            quads = np.stack(quads_list, axis=0).astype(np.float32)           # (N,4,2)
            kpts = np.asarray(kpts_list, dtype=np.float32)                    # (N,2)
            labels = np.asarray(labels_list, dtype=np.int64)                  # (N,)
            # provisional angles (filled after transform from quads again)
            angles = np.zeros((len(quads_list),), dtype=np.float32)

        # -- build sample for the transform pipeline
        sample = {
            "image": img,               # H,W,3 RGB
            "boxes": boxes,             # (N,4) AABB (pixels)
            "quads": quads,             # (N,4,2) polygon (pixels)
            "kpts": kpts,               # (N,2) (pixels)
            "labels": labels,           # (N,)
            "angles": angles,           # place-holder; we recompute from quads
            "path": img_path,
            "meta": {"orig_size": (H0, W0), "path": img_path},
        }

        # -- apply your project's transform pipeline
        sample = transform_sample(sample, self.img_size)
        # Expected to preserve keys above in pixel space.

        # --- ensure types are tensors (collate relies on this) ---
        # image
        if not torch.is_tensor(sample["image"]):
            # Expect transform_sample to have made it a CHW float tensor already.
            # If not, do a minimal conversion:
            im = sample["image"]
            if isinstance(im, np.ndarray):
                im = torch.from_numpy(im.transpose(2, 0, 1)).float()  # HWC->CHW
            sample["image"] = im

        # labels
        if not torch.is_tensor(sample.get("labels", None)):
            sample["labels"] = torch.as_tensor(sample.get("labels", []), dtype=torch.long)
        # quads
        if not torch.is_tensor(sample.get("quads", None)):
            sample["quads"] = torch.as_tensor(sample.get("quads", np.zeros((0, 4, 2), np.float32)),
                                              dtype=torch.float32)
        # boxes (AABB)
        if not torch.is_tensor(sample.get("boxes", None)):
            sample["boxes"] = torch.as_tensor(sample.get("boxes", np.zeros((0, 4), np.float32)),
                                              dtype=torch.float32)
        # kpts
        if not torch.is_tensor(sample.get("kpts", None)):
            sample["kpts"] = torch.as_tensor(sample.get("kpts", np.zeros((0, 2), np.float32)),
                                             dtype=torch.float32)

        # --- recompute OBB from quads after transforms ---
        quads_t: torch.Tensor = sample["quads"]  # (N,4,2) in pixels
        if quads_t.numel() > 0:
            quads_t = quads_t.view(-1, 4, 2)
            obb_list = []
            ang_list = []
            for i in range(quads_t.shape[0]):
                cx, cy, ww, hh, th = quad_to_cxcywh_angle(quads_t[i])  # theta (radians)
                # avoid zero-size
                ww = float(max(ww, 1.0))
                hh = float(max(hh, 1.0))
                obb_list.append([float(cx), float(cy), ww, hh, float(th)])
                ang_list.append(float(th))
            bboxes = torch.tensor(obb_list, dtype=torch.float32)
            sample["angles"] = torch.tensor(ang_list, dtype=torch.float32)
        else:
            bboxes = torch.zeros((0, 5), dtype=torch.float32)
            sample["angles"] = torch.zeros((0,), dtype=torch.float32)

        sample["bboxes"] = bboxes  # (N,5) in pixels, theta in radians

        # --- build normalized Nx11 'targets' for any legacy paths ---
        # Format: [cls + 8*poly + 2*kpt], all normalized to current W,H
        # Use *post-transform* image size
        img_t: torch.Tensor = sample["image"]
        if img_t.dim() == 3:
            _, Hc, Wc = img_t.shape
        else:
            # Should never happen for a single sample
            Hc, Wc = self.img_size, self.img_size

        quads_n = sample["quads"]
        kpts_n = sample["kpts"]
        labels_n = sample["labels"]

        if quads_n.numel() > 0:
            # normalize to [0,1]
            qn = quads_n.clone()
            qn[..., 0] /= float(Wc)
            qn[..., 1] /= float(Hc)
            qn = torch.clamp(qn, 0.0, 1.0)

            kn = kpts_n.clone()
            if kn.numel() > 0:
                kn[:, 0] = torch.clamp(kn[:, 0] / float(Wc), 0.0, 1.0)
                kn[:, 1] = torch.clamp(kn[:, 1] / float(Hc), 0.0, 1.0)
            else:
                kn = torch.zeros((qn.shape[0], 2), dtype=torch.float32)

            cls_f = labels_n.to(torch.float32).view(-1, 1)
            flat_poly = qn.view(qn.shape[0], 8)  # (N,8)
            targets = torch.cat([cls_f, flat_poly, kn], dim=1)  # (N,11)
        else:
            targets = torch.zeros((0, 11), dtype=torch.float32)

        sample["targets"] = targets

        # --- assertions & sanity checks ---
        assert torch.is_tensor(sample["image"]) and sample["image"].dim() == 3, "image must be CHW tensor"
        assert all(k in sample for k in ("bboxes", "labels", "kpts", "quads", "boxes", "targets")), \
            "dataset sample missing required keys"
        for name in ("bboxes", "labels", "kpts", "quads", "boxes", "targets"):
            t = sample[name]
            if torch.is_tensor(t):
                if torch.isnan(t).any():
                    raise ValueError(f"NaN in sample[{name}] for {img_path}")

        # Unify naming expected by collate/trainer
        if "path" in sample and "paths" not in sample:
            sample["paths"] = sample["path"]

        # enrich meta with current size
        meta = sample.get("meta", {})
        meta["curr_size"] = (int(Hc), int(Wc))
        sample["meta"] = meta

        return sample
