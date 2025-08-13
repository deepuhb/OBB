# src/models/layers/rot_roi.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def obb_to_affine(cx, cy, w, h, angle_deg, out_size, expand=1.0, dtype=None, device=None):
    """Build 2x3 affine that maps output (crop) -> input (feature map) coordinates."""
    device = device if device is not None else (cx.device if isinstance(cx, torch.Tensor) else "cpu")
    dtype = dtype if dtype is not None else (cx.dtype if isinstance(cx, torch.Tensor) else torch.float32)
    s = float(expand)
    w_s = w * s
    h_s = h * s
    th = angle_deg * (math.pi / 180.0)
    c = torch.cos(th)
    s_ = torch.sin(th)
    # scale from crop pixels (out_size) to input pixels along x,y
    sx = w_s / float(out_size)
    sy = h_s / float(out_size)
    # matrix that maps crop -> input
    M = torch.tensor([[ c*sx, -s_*sy, cx],
                      [ s*sx,  c*sy, cy]], dtype=dtype, device=device)  # NOTE: s refers to sin; shadow name
    return M

def affine_grid_sample(input, M, out_size):
    """Vectorized grid_sample for a batch of transforms.
    input: (N,C,H,W), M: (N,2,3) mapping crop->input, out: (N,C,S,S)"""
    N, C, H, W = input.shape
    grid = F.affine_grid(M, size=(N, C, out_size, out_size), align_corners=False)  # grid in [-1,1]
    out = F.grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out

class RotatedROIPool(nn.Module):
    """Differentiable rotated cropper for FPN feature maps."""
    def __init__(self, out_size=64, expand=1.25, feat_down=8):
        super().__init__()
        self.S = int(out_size)
        self.expand = float(expand)
        self.feat_down = int(feat_down)  # downsample factor of the chosen FPN level

    def forward(self, feat, obb_list):
        """
        feat: (B,C,Hf,Wf) feature map (e.g., from P3)
        obb_list: list of (Ni,5) [cx,cy,w,h,deg] in IMAGE PIXELS (not feature coords)
        Returns crops: (sum_i Ni, C, S, S), metas (list of dict: bix, M)
        """
        device = feat.device
        _, C, Hf, Wf = feat.shape
        Ms = []
        idxs = []
        metas = []
        scale = 1.0 / float(self.feat_down)
        for bix, obb in enumerate(obb_list):
            if obb is None or len(obb) == 0:
                continue
            obb = obb.to(device)
            for j in range(obb.shape[0]):
                cx, cy, w, h, deg = obb[j]
                M = obb_to_affine(cx*scale, cy*scale, w*scale, h*scale, deg, self.S, self.expand,
                                  dtype=feat.dtype, device=device).unsqueeze(0)  # (1,2,3)
                Ms.append(M)
                idxs.append(bix)
                metas.append({"bix": bix, "M": M[0]})
        if not Ms:
            return torch.empty(0, C, self.S, self.S, device=device, dtype=feat.dtype), metas
        Ms = torch.cat(Ms, dim=0)                      # (N,2,3)
        src = feat[idxs, ...]                          # (N,C,Hf,Wf)
        crops = affine_grid_sample(src, Ms, self.S)    # (N,C,S,S)
        return crops, metas

def invert_affine_2x2(A, b):
    """Given x' = A x + b, return A^{-1}, b' so that x = A^{-1}(x' - b)."""
    Ainv = torch.inverse(A)
    return Ainv, -Ainv @ b
