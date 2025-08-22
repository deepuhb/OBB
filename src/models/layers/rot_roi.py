# src/models/layers/rot_roi.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_obb_2d(obb: torch.Tensor, *, device, dtype) -> torch.Tensor:
    """
    Coerce an OBB tensor/array/list into shape (N,5) on the given device/dtype.
    Accepts (5,), (N,5), or anything that can be reshaped to (-1,5).
    Extra columns are sliced off; rows with <5 columns are dropped.
    NaN/Inf rows are removed.
    """
    obb = torch.as_tensor(obb, device=device, dtype=dtype)

    # Make at least 2D
    if obb.ndim == 1:
        obb = obb.unsqueeze(0)
    elif obb.ndim > 2:
        obb = obb.reshape(-1, obb.shape[-1])

    # Enforce 5 columns
    if obb.shape[-1] < 5:
        # Not enough columns -> nothing usable
        return torch.empty((0, 5), device=device, dtype=dtype)
    if obb.shape[-1] > 5:
        obb = obb[..., :5]

    # Drop invalid rows (NaN/Inf)
    if obb.numel():
        valid = torch.isfinite(obb).all(dim=1)
        if valid.dim() == 0:
            # single row case -> scalar bool
            valid = valid.unsqueeze(0)
        obb = obb[valid]

    return obb


def obb_to_affine(
    obb: torch.Tensor,  # (N,5) [cx,cy,w,h,deg] in IMAGE pixels
    S: int,             # crop size (e.g., 64)
    expand: float,      # enlarge factor (e.g., 1.25)
    feat_down: int,     # stride of the feature map (e.g., 8 for P3)
    Hf: int, Wf: int,   # feature map height/width
):
    """
    Returns:
      theta: (N,2,3) for F.affine_grid (normalized coords)
      M    : (N,2,3) mapping crop pixel (u,v,1) -> FEATURE pixel (x_feat,y_feat)
    """
    # Be robust to 1-D or odd shapes
    obb = _ensure_obb_2d(obb, device=obb.device, dtype=obb.dtype)
    device = obb.device
    if obb.numel() == 0:
        empty = torch.zeros((0, 2, 3), device=device, dtype=obb.dtype)
        return empty, empty

    cx, cy, w, h, deg = obb[:, 0], obb[:, 1], obb[:, 2], obb[:, 3], obb[:, 4]

    # to feature coords + expand
    cx_f = cx / feat_down
    cy_f = cy / feat_down
    w_f = (w.clamp_min(1.0) * expand) / feat_down
    h_f = (h.clamp_min(1.0) * expand) / feat_down
    ang = torch.deg2rad(deg)

    # normalized scales/translate for affine_grid
    sx = (w_f / Wf) * 2.0
    sy = (h_f / Hf) * 2.0
    tx = (cx_f / Wf) * 2.0 - 1.0
    ty = (cy_f / Hf) * 2.0 - 1.0

    cos = torch.cos(ang)
    sin = torch.sin(ang)
    a11 = cos * sx
    a12 = -sin * sy
    a13 = tx
    a21 = sin * sx
    a22 = cos * sy
    a23 = ty
    theta = torch.stack(
        [torch.stack([a11, a12, a13], dim=-1), torch.stack([a21, a22, a23], dim=-1)],
        dim=1,
    )  # (N,2,3)

    # M: crop pixel -> FEATURE pixel (not normalized)
    A = cos * (w_f / S)
    Bv = -sin * (h_f / S)
    A2 = sin * (w_f / S)
    B2 = cos * (h_f / S)
    Cx = cx_f - 0.5 * S * (A + Bv)
    Cy = cy_f - 0.5 * S * (A2 + B2)
    M = torch.stack(
        [torch.stack([A, Bv, Cx], dim=-1), torch.stack([A2, B2, Cy], dim=-1)], dim=1
    )  # (N,2,3)

    return theta, M


class RotatedROIPool(nn.Module):
    """
    Memory-safe rotated ROI cropper for top-down keypoint head.

    - Caps ROIs per image: topk + score_thresh
    - Chunks grid_sample calls (roi_chunk)
    - Uses .expand() view to avoid copying feature maps per ROI
    - Returns:
        crops: (N_total, C, S, S)
        metas: list of dicts {'bix': int, 'M': (2,3), 'oi': int}
    """
    def __init__(self, out_size: int = 64, expand: float = 1.25, feat_down: int = 8):
        super().__init__()
        self.S = int(out_size)
        self.outsize = int(out_size)  # kept for BC
        self.expand = float(expand)
        self.feat_down = int(feat_down)  # stride of the feature we sample (e.g., 8 for P3)

    @torch.no_grad()
    def forward(
        self,
        feat: torch.Tensor,          # (B,C,Hf,Wf)
        obb_list,                    # list length B: each (Ni,5) [cx,cy,w,h,deg] in image pixels; accepts (5,) too
        scores_list=None,            # list length B: each (Ni,) or None
        topk: int = None,
        chunk: int = 128,
        score_thresh: float = None,
    ):
        device = feat.device
        dtype = feat.dtype
        B, C, Hf, Wf = feat.shape
        S = self.S

        crops = []
        metas = []

        for b in range(B):
            # ---- Robust OBB normalization to (N,5) ----
            obb_raw = obb_list[b]
            if obb_raw is None:
                continue

            obb = _ensure_obb_2d(obb_raw, device=device, dtype=dtype)
            if obb.numel() == 0:
                continue

            # Align scores (optional) to obb length
            scores = None
            if scores_list is not None:
                scr = scores_list[b]
                if scr is not None:
                    scores = torch.as_tensor(scr, device=device, dtype=dtype).reshape(-1)
                    if scores.shape[0] != obb.shape[0]:
                        m = min(int(scores.shape[0]), int(obb.shape[0]))
                        if m == 0:
                            continue
                        obb = obb[:m]
                        scores = scores[:m]

            idx_all = torch.arange(obb.shape[0], device=device)

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

            # ---- Compute theta, M (normalized and pixel-space maps) ----
            cx = obb[:, 0]; cy = obb[:, 1]; w = obb[:, 2]; h = obb[:, 3]; deg = obb[:, 4]
            cx_f = cx / self.feat_down
            cy_f = cy / self.feat_down
            w_f = (w.clamp_min(1.0) * self.expand) / self.feat_down
            h_f = (h.clamp_min(1.0) * self.expand) / self.feat_down
            ang = torch.deg2rad(deg)

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

            # Pixel-space crop->feature map transform (for back-projection of UV)
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

            # ---- Chunked sampling (memory-safe) ----
            for s in range(0, N, int(chunk)):
                e = min(s + int(chunk), N)
                th = theta[s:e]
                grid = F.affine_grid(th, size=(e - s, C, S, S), align_corners=False)
                src = feat[b:b+1].expand(e - s, -1, -1, -1)  # view, no copy
                cr = F.grid_sample(src, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                crops.append(cr)
                # Keep a compact meta per ROI
                for j in range(s, e):
                    metas.append({"bix": b, "M": M_all[j].detach(), "oi": int(idx_all[j])})

        if len(crops) == 0:
            return feat.new_zeros((0, C, S, S)), []
        return torch.cat(crops, dim=0), metas
