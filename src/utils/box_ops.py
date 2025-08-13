import torch
import math

def quad_to_cxcywh_angle(quad_xy):
    # quad_xy: Tensor (4,2) absolute pixel coords in order (x1,y1..x4,y4)
    # returns: cx,cy,w,h,theta (theta radians)
    p = quad_xy
    cx = p[:,0].mean()
    cy = p[:,1].mean()
    w = torch.linalg.vector_norm(p[1]-p[0])
    h = torch.linalg.vector_norm(p[2]-p[1])
    theta = torch.atan2(p[1,1]-p[0,1], p[1,0]-p[0,0])
    if h > w:
        w, h = h, w
        theta = theta + math.pi/2
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
