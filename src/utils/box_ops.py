import torch
import math

def quad_to_cxcywh_angle(quad_xy):
    # quad_xy: Tensor (4,2) absolute pixel coords in order (x1,y1..x4,y4)
    # returns: cx,cy,w,h,theta (theta radians)
    p = torch.as_tensor(quad_xy, dtype=torch.float32)
    cx = float(p[:, 0].mean())
    cy = float(p[:, 1].mean())
    e01 = p[1] - p[0]
    e12 = p[2] - p[1]
    w = float(torch.linalg.vector_norm(e01))
    h = float(torch.linalg.vector_norm(e12))
    theta = math.atan2(float(e01[1]), float(e01[0]))  # radians in (-π, π]
    if h > w:
        w, h = h, w
        theta += math.pi / 2.0
    # wrap to [-π/2, π/2)
    while theta < -math.pi / 2: theta += math.pi
    while theta >= math.pi / 2: theta -= math.pi
    return cx, cy, w, h, theta

def bbox_iou_xyxy(a: torch.Tensor, b: torch.Tensor):
    tl = torch.maximum(a[..., :2], b[..., :2])
    br = torch.minimum(a[..., 2:], b[..., 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0]*wh[..., 1]
    area_a = (a[..., 2]-a[..., 0]).clamp(min=0)*(a[..., 3]-a[..., 1]).clamp(min=0)
    area_b = (b[..., 2]-b[..., 0]).clamp(min=0)*(b[..., 3]-b[..., 1]).clamp(min=0)
    union = (area_a + area_b - inter).clamp(min=1e-9)
    return inter/union
