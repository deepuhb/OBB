# src/models/layers/rot_roi.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def obb_to_affine(
    obb: torch.Tensor,           # (N,5) [cx,cy,w,h,deg] in IMAGE pixels
    S: int,                      # crop size (e.g., 64)
    expand: float,               # enlarge factor (e.g., 1.25)
    feat_down: int,              # stride of the feature map (e.g., 8 for P3)
    Hf: int, Wf: int,            # feature map height/width
):
    """
    Returns:
      theta: (N,2,3) for F.affine_grid (normalized coords)
      M    : (N,2,3) mapping crop pixel (u,v,1) -> FEATURE pixel (x_feat,y_feat)
    """
    device = obb.device
    if obb.numel() == 0:
        empty = torch.zeros((0,2,3), device=device, dtype=obb.dtype)
        return empty, empty

    cx, cy, w, h, deg = obb[:,0], obb[:,1], obb[:,2], obb[:,3], obb[:,4]
    # to feature coords + expand
    cx_f = cx / feat_down
    cy_f = cy / feat_down
    w_f  = (w.clamp_min(1.0) * expand) / feat_down
    h_f  = (h.clamp_min(1.0) * expand) / feat_down
    ang  = torch.deg2rad(deg)

    # normalized scales/translate for affine_grid
    sx = (w_f / Wf) * 2.0
    sy = (h_f / Hf) * 2.0
    tx = (cx_f / Wf) * 2.0 - 1.0
    ty = (cy_f / Hf) * 2.0 - 1.0

    cos = torch.cos(ang); sin = torch.sin(ang)
    a11 =  cos * sx; a12 = -sin * sy; a13 = tx
    a21 =  sin * sx; a22 =  cos * sy; a23 = ty
    theta = torch.stack([
                torch.stack([a11, a12, a13], dim=-1),
                torch.stack([a21, a22, a23], dim=-1)
             ], dim=1)  # (N,2,3)

    # M: crop pixel -> FEATURE pixel (not normalized)
    # x_feat = A*u + B*v + Cx ; y_feat = A2*u + B2*v + Cy  for u,v in [0,S)
    A  =  cos * (w_f / S)
    Bv = -sin * (h_f / S)
    A2 =  sin * (w_f / S)
    B2 =  cos * (h_f / S)
    Cx = cx_f - 0.5 * S * (A + Bv)
    Cy = cy_f - 0.5 * S * (A2 + B2)
    M = torch.stack([
            torch.stack([A,  Bv, Cx], dim=-1),
            torch.stack([A2, B2, Cy], dim=-1)
        ], dim=1)  # (N,2,3)

    return theta, M


class RotatedROIPool(nn.Module):
    """
    Memory-safe rotated ROI cropper for top-down keypoint head.

    - Caps ROIs per image: topk + score_thresh
    - Chunks grid_sample calls (roi_chunk)
    - Uses .expand() view to avoid copying feature maps per ROI
    - Returns:
        crops: (N_total, C, S, S)
        metas: list of dicts with {'bix': int, 'M': (2,3) mapping crop px -> feat px}
    """
    def __init__(self, out_size: int = 64, expand: float = 1.25, feat_down: int = 8):
        super().__init__()
        # Keep both names for compatibility; use self.S everywhere else
        self.S = int(out_size)
        self.outsize = int(out_size)
        self.expand = float(expand)
        self.feat_down = int(feat_down)  # stride of the feature we sample (e.g., 8 for P3)

    @torch.no_grad()
    def forward(
        self,
        feat: torch.Tensor,          # (B,C,Hf,Wf)
        obb_list,                    # list of length B, each (Ni,5) [cx,cy,w,h,deg] in image pixels (Tensor) or None
        scores_list=None,            # list of length B, each (Ni,) scores (Tensor) or None
        topk: int = None,
        chunk: int = 128,
        score_thresh: float = None,
    ):
        device = feat.device
        B, C, Hf, Wf = feat.shape
        S = self.S  # <- defined in __init__

        crops = []
        metas = []

        for b in range(B):
            obb = obb_list[b]
            if obb is None or (torch.is_tensor(obb) and obb.numel() == 0):
                continue

            if not torch.is_tensor(obb):
                # allow list-of-lists; convert if needed
                obb = torch.tensor(obb, device=device, dtype=feat.dtype)

            idx_all = torch.arange(obb.shape[0], device=device)
            scores = scores_list[b] if scores_list is not None else None

            # 1) score filter
            if scores is not None and score_thresh is not None:
                keep = scores >= float(score_thresh)
                if keep.any():
                    obb = obb[keep]
                    idx_all = idx_all[keep]
                    scores = scores[keep]
                else:
                    continue

            # 2) top-k cap
            if topk is not None and obb.shape[0] > int(topk):
                if scores is not None:
                    sel = torch.topk(scores, k=int(topk), largest=True).indices
                else:
                    sel = torch.arange(int(topk), device=device)
                obb = obb[sel]
                idx_all = idx_all[sel]

            N = int(obb.shape[0])
            if N == 0:
                continue

            # Convert to feature coordinates and expand ROI
            cx_f = obb[:, 0] / self.feat_down
            cy_f = obb[:, 1] / self.feat_down
            w_f  = (obb[:, 2].clamp_min(1.0) * self.expand) / self.feat_down
            h_f  = (obb[:, 3].clamp_min(1.0) * self.expand) / self.feat_down
            ang  = torch.deg2rad(obb[:, 4])

            # Normalized affine params (grid expects normalized coords in [-1,1])
            sx = (w_f / Wf) * 2.0
            sy = (h_f / Hf) * 2.0
            tx = (cx_f / Wf) * 2.0 - 1.0
            ty = (cy_f / Hf) * 2.0 - 1.0

            cos = torch.cos(ang)
            sin = torch.sin(ang)
            a11 =  cos * sx; a12 = -sin * sy; a13 = tx
            a21 =  sin * sx; a22 =  cos * sy; a23 = ty
            theta = torch.stack([
                        torch.stack([a11, a12, a13], dim=-1),
                        torch.stack([a21, a22, a23], dim=-1)
                     ], dim=1)  # (N,2,3)

            # Map crop px (uS,vS,1) -> feature px (x_feat,y_feat)
            A  =  cos * (w_f / S)
            Bv = -sin * (h_f / S)
            A2 =  sin * (w_f / S)
            B2 =  cos * (h_f / S)
            Cx = cx_f - 0.5 * S * (A + Bv)
            Cy = cy_f - 0.5 * S * (A2 + B2)
            M_all = torch.stack([
                        torch.stack([A,  Bv, Cx], dim=-1),
                        torch.stack([A2, B2, Cy], dim=-1)
                    ], dim=1)  # (N,2,3)

            # Chunked sampling to bound memory (expand() creates a view, no copy)
            for s in range(0, N, int(chunk)):
                e = min(s + int(chunk), N)
                th = theta[s:e]
                grid = F.affine_grid(th, size=(e - s, C, S, S), align_corners=False)
                src = feat[b:b+1].expand(e - s, -1, -1, -1)  # view over the same feature map
                cr = F.grid_sample(src, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                crops.append(cr)
                for j in range(s, e):
                    metas.append({"bix": b, "M": M_all[j].detach(), "oi": int(j)})

        if len(crops) == 0:
            return feat.new_zeros((0, C, S, S)), []
        return torch.cat(crops, dim=0), metas