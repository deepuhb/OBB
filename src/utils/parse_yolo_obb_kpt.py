# src/data/utils/parse_yolo_obb_kpt.py
import math
import numpy as np
import torch

def _denorm_xy(x, y, W, H):
    return float(x) * W, float(y) * H

def _poly4_to_obb_xywha_deg(pxy: np.ndarray):
    # pxy: (4,2) in pixels, ordered around box
    (x1,y1), (x2,y2), (x3,y3), (x4,y4) = pxy.astype(np.float32)
    cx = (x1 + x2 + x3 + x4) * 0.25
    cy = (y1 + y2 + y3 + y4) * 0.25
    w  = float(np.hypot(x2 - x1, y2 - y1))
    h  = float(np.hypot(x3 - x2, y3 - y2))
    ang_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
    return cx, cy, max(w,1.0), max(h,1.0), ang_deg

def parse_yolo11_obb_kpt_lines(lines, W: int, H: int):
    """
    lines: iterable of 'cls x1 y1 x2 y2 x3 y3 x4 y4 kx ky' (normalized 0..1)
    returns:
      labels  : LongTensor [N]
      bboxes  : FloatTensor [N,5] (cx,cy,w,h,ang_radians) in pixels
      kpts    : FloatTensor [N,2] (kx,ky) in pixels
    """
    labels, obbs, kpts = [], [], []
    for ln in lines:
        vals = [float(v) for v in ln.strip().split()]
        if len(vals) < 11:
            continue
        cls = int(vals[0])
        poly = np.array(vals[1:9], dtype=np.float32).reshape(4,2)
        kxy  = np.array(vals[9:11], dtype=np.float32)

        # de-normalize
        pts_px = np.stack([_denorm_xy(x, y, W, H) for x,y in poly], axis=0)
        kx, ky = _denorm_xy(kxy[0], kxy[1], W, H)

        cx, cy, w, h, ang_deg = _poly4_to_obb_xywha_deg(pts_px)

        labels.append(cls)
        obbs.append([cx, cy, w, h, math.radians(ang_deg)])  # radians for the loss
        kpts.append([kx, ky])

    if len(labels) == 0:
        return (torch.zeros((0,),  dtype=torch.long),
                torch.zeros((0,5), dtype=torch.float32),
                torch.zeros((0,2), dtype=torch.float32))
    return (torch.tensor(labels, dtype=torch.long),
            torch.tensor(obbs,   dtype=torch.float32),
            torch.tensor(kpts,   dtype=torch.float32))
