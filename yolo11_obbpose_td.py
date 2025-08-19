# src/models/yolo11_obbpose_td.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.rot_roi import RotatedROIPool


# ---- Tiny YOLO11-ish blocks (you can swap with your existing backbone/neck) ----
class Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        p = k//2 if p is None else p
        self.cv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.cv(x)))

class C2f(nn.Module):
    """C2f (simplified)"""
    def __init__(self, c1, c2, n=2, shortcut=False):
        super().__init__()
        c_ = int(c2//2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[Conv(c_, c_, 3, 1) for _ in range(n)])
        self.cv3 = Conv(2*c_, c2, 1, 1)
        self.sc = shortcut
    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1,y2], 1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_*4, c2, 1, 1)
        self.k = k
    def forward(self, x):
        x = self.cv1(x)
        y1 = F.max_pool2d(x, self.k, stride=1, padding=self.k//2)
        y2 = F.max_pool2d(y1, self.k, stride=1, padding=self.k//2)
        y3 = F.max_pool2d(y2, self.k, stride=1, padding=self.k//2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

class Backbone(nn.Module):
    def __init__(self, c=3, w=0.5, d=0.33):
        super().__init__()
        c1, c2, c3, c4, c5 = int(64*w), int(128*w), int(256*w), int(512*w), int(512*w)
        n = max(1, int(3*d))
        self.stem = Conv(c, c1, 3, 2)
        self.c2 = nn.Sequential(Conv(c1, c1, 3, 1), Conv(c1, c1, 3, 1))
        self.p3 = nn.Sequential(Conv(c1, c2, 3, 2), C2f(c2, c2, n))
        self.p4 = nn.Sequential(Conv(c2, c3, 3, 2), C2f(c3, c3, n))
        self.p5 = nn.Sequential(Conv(c3, c4, 3, 2), C2f(c4, c4, n), SPPF(c4, c5))
    def forward(self, x):
        x = self.stem(x)
        x = self.c2(x)
        p3 = self.p3(x)   # /8
        p4 = self.p4(p3)  # /16
        p5 = self.p5(p4)  # /32
        return p3, p4, p5


class FPNPAN(nn.Module):
    """
    Inputs:  p3 (c2), p4 (c3), p5 (c5)
    Outputs: n3 (c2), d4 (c3), d5 (c5)  -> strides [8,16,32]
    All concatenations are sized to match exactly.
    """
    def __init__(self, in_ch):  # in_ch = [p3_ch, p4_ch, p5_ch]
        super().__init__()
        c2, c3, c5 = in_ch

        # ---- top-down FPN ----
        # lateral from p5 -> c3 to match p4 width
        self.lat5_to_4 = Conv(c5, c3, 1, 1)
        # fuse up(p5->c3) with p4(c3)
        self.fuse4 = Conv(c3 + c3, c3, 3, 1)

        # reduce fused c3 to c2 for next upsample stage
        self.red4_to_3 = Conv(c3, c2, 1, 1)
        # lateral p3->c2 already
        # fuse up(red4_to_3)->c2 with p3(c2)
        self.fuse3 = Conv(c2 + c2, c2, 3, 1)

        # ---- bottom-up PAN ----
        # down from n3(c2) to c3
        self.down4 = Conv(c2, c3, 3, 2)
        # fuse down(n3)->c3 with n4(c3)
        self.fuse4d = Conv(c3 + c3, c3, 3, 1)

        # down from d4(c3) to c5
        self.down5 = Conv(c3, c5, 3, 2)
        # fuse down(d4)->c5 with p5(c5)
        self.fuse5d = Conv(c5 + c5, c5, 3, 1)

    def forward(self, p3, p4, p5):
        # top-down
        n4 = torch.cat([F.interpolate(self.lat5_to_4(p5), scale_factor=2, mode="nearest"), p4], dim=1)
        n4 = self.fuse4(n4)                 # (B,c3,H/16,W/16)

        n3 = torch.cat([F.interpolate(self.red4_to_3(n4), scale_factor=2, mode="nearest"), p3], dim=1)
        n3 = self.fuse3(n3)                 # (B,c2,H/8,W/8)

        # bottom-up
        d4 = self.fuse4d(torch.cat([self.down4(n3), n4], dim=1))  # (B,c3,H/16,W/16)
        d5 = self.fuse5d(torch.cat([self.down5(d4), p5], dim=1))  # (B,c5,H/32,W/32)

        return n3, d4, d5  # strides [8,16,32]


# ---- Heads ----
class OBBHead(nn.Module):
    """Anchor-free YOLO head predicting (tx,ty,tw,th,sin,cos,obj,cls[...]) per location."""
    def __init__(self, ch, nc):
        super().__init__()
        self.nc = nc
        self.m = nn.ModuleList()
        for c in ch:
            self.m.append(nn.Sequential(
                Conv(c, c, 3, 1), nn.Conv2d(c, 7+nc, 1, 1)  # 7: tx,ty,tw,th,sin,cos,obj
            ))
    def forward(self, feats):
        return [h(f) for h, f in zip(self.m, feats)]  # list of (B,7+nc,H,W)

class KptTDHead(nn.Module):
    """Top-down keypoint head that runs on rotated crops from P3 features."""
    def __init__(self, in_ch=256, S=64):
        super().__init__()
        base = 64
        self.net = nn.Sequential(
            Conv(in_ch, base, 3, 1), nn.MaxPool2d(2),
            Conv(base, base*2, 3, 1), nn.MaxPool2d(2),
            Conv(base*2, base*4, 3, 1), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base*4, 64), nn.SiLU(),
            nn.Linear(64, 2), nn.Sigmoid()  # (u,v) in [0,1]
        )
    def forward(self, crops):
        return self.net(crops)  # (N,2)


class YOLO11_OBBPOSE_TD(nn.Module):
    def __init__(self, num_classes=1, width=0.5, depth=0.33, kpt_crop=64, kpt_expand=1.25,
                 kpt_topk=128, roi_chunk=128, score_thresh_kpt=0.25):
        super().__init__()
        self.backbone = Backbone(w=width, d=depth)

        # Channel plan that **matches Backbone** outputs:
        # p3 -> c2, p4 -> c3, p5 -> c5
        c2 = int(128 * width)   # p3 channels
        c3 = int(256 * width)   # p4 channels
        c5 = int(512 * width)   # p5 channels (after SPPF)

        # Neck consumes (p3_c, p4_c, p5_c) in that order
        self.neck = FPNPAN([c2, c3, c5])

        # Neck outputs [n3(c2), d4(c3), d5(c5)]
        self.det_head = OBBHead([c2, c3, c5], nc=num_classes)

        # Top-down keypoint crops come from P3-like feature (n3), stride=8
        self.roi = RotatedROIPool(out_size=kpt_crop, expand=kpt_expand, feat_down=8)
        self.kpt_head = KptTDHead(in_ch=c2, S=kpt_crop)
        # stash for kpt_from_obbs
        self.kpt_topk = int(kpt_topk)
        self.roi_chunk = int(roi_chunk)
        self.score_thresh_kpt = float(score_thresh_kpt)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)       # (c2, c3, c5)
        n3, d4, d5 = self.neck(p3, p4, p5)  # (c2, c3, c5)
        det = self.det_head([n3, d4, d5])
        return {"det": det, "feats": [n3, d4, d5]}

    def _select_kpt_feat(self, feats):
        # robustly pick a single feature map (finest resolution)
        if torch.is_tensor(feats):
            return feats
        if isinstance(feats, (list, tuple)):
            feats = [f for f in feats if torch.is_tensor(f)]
            return max(feats, key=lambda t: t.shape[-1] * t.shape[-2])
        if isinstance(feats, dict):
            for k in ("kpt", "p3", "P3"):
                v = feats.get(k)
                if torch.is_tensor(v):
                    return v
            feats = [v for v in feats.values() if torch.is_tensor(v)]
            return max(feats, key=lambda t: t.shape[-1] * t.shape[-2])
        raise TypeError(f"Unsupported feats type: {type(feats)}")

    @torch.no_grad()
    def export_decode(
            self,
            det_maps_or_dense: torch.Tensor,
            imgs: torch.Tensor,
            score_thr: float = 0.25,
            max_det: int = 300,
    ):
        """
        Unified decoder for export:
          - If input is list[Tensor(B,C,Hf,Wf)]: route to decode_obb_from_pyramids().
          - If input is Tensor(B,N,K) with K>=7 and channel order
                [cx, cy, w, h, sinθ, cosθ, obj, cls...]
            (or same order but with *logits* that still need sigmoid for obj/cls)
            decode to per-image dicts:
                {'boxes': (Ni,5)[cx,cy,w,h,theta(rad)], 'scores': (Ni,), 'cls': Optional[(Ni,)]}
        """
        import torch

        # Case A: pyramids (training/eval)
        if isinstance(det_maps_or_dense, (list, tuple)):
            return self.decode_obb_from_pyramids(det_maps_or_dense, imgs, score_thr=score_thr, max_det=max_det)

        # Case B: dense predictions (B,N,K)
        t = det_maps_or_dense
        if (not torch.is_tensor(t)) or t.dim() != 3 or t.size(-1) < 7:
            # graceful fallback: nothing to decode
            B = imgs.shape[0]
            z = imgs.new_zeros
            return [{"boxes": z((0, 5)), "scores": z((0,)), "cls": None} for _ in range(B)]

        B, N, K = t.shape
        device = t.device

        # geometry
        cx = t[..., 0]
        cy = t[..., 1]
        w = t[..., 2]
        h = t[..., 3]
        s = t[..., 4]  # sinθ (raw)
        c = t[..., 5]  # cosθ (raw)
        ang = torch.atan2(s, c)  # radians

        # scores & classes
        obj = t[..., 6].sigmoid()
        cls = None
        if K > 7:
            cls_logits = t[..., 7:]
            cls_prob = cls_logits.sigmoid()  # (B,N,nc)
            cls_score, cls_idx = cls_prob.max(dim=-1)  # (B,N)
            scores = obj * cls_score
            cls = cls_idx
        else:
            scores = obj

        # threshold & top-k per image
        out = []
        for b in range(B):
            sb = scores[b]
            keep = sb >= float(score_thr)
            if keep.any():
                bx = torch.stack([cx[b][keep], cy[b][keep], w[b][keep], h[b][keep], ang[b][keep]], dim=-1)
                sc = sb[keep]
                if bx.shape[0] > max_det:
                    idx = torch.topk(sc, k=max_det, largest=True).indices
                    bx, sc = bx[idx], sc[idx]
                    clb = cls[b][keep][idx] if cls is not None else None
                else:
                    clb = cls[b][keep] if cls is not None else None
            else:
                bx = imgs.new_zeros((0, 5), device=device)
                sc = imgs.new_zeros((0,), device=device)
                clb = None

            out.append({"boxes": bx, "scores": sc, "cls": clb})
        return out

    @torch.no_grad()
    
    def decode_obb_from_pyramids(
            self,
            pyramids,
            imgs: torch.Tensor,
            score_thr: float = 0.25,
            max_det: int = 300,
            use_nms: bool = True,
            nms_iou_thr: float = 0.5,
            pre_topk: int = 2000,
    ):
        """
        Decode anchor-free head outputs with channel order:
          [tx, ty, tw, th, sinθ, cosθ, obj, cls...]
        Returns per-image dictionaries containing BOTH:
          - legacy keys: {"boxes": (Ni,5)[cx,cy,w,h,theta(rad)], "scores": (Ni,), "cls": Optional[(Ni,)]}
          - extended keys: "obb" (degrees), "labels" (same as "cls" but always LongTensor), "polygons" (list[Tensor(4,2)])
        Angle unit:
          - boxes[..., 4] is radians (legacy compatibility)
          - obb[..., 4]   is degrees (extended convenience)
        """
        import torch, math

        assert isinstance(pyramids, (list, tuple)) and len(pyramids) > 0, "pyramids must be list/tuple of Tensors"
        B, _, H, W = imgs.shape
        device = imgs.device

        # --- Helper functions (local) ---
        def _obb_corners(cx, cy, w, h, ang_rad):
            """Return (4,2) corners in image coords for a single OBB."""
            c = torch.cos(ang_rad); s = torch.sin(ang_rad)
            wx = w / 2.0; hy = h / 2.0
            pts = torch.tensor([[-wx, -hy],
                                [ wx, -hy],
                                [ wx,  hy],
                                [-wx,  hy]], device=device, dtype=torch.float32)
            R = torch.tensor([[c, -s],
                              [s,  c]], device=device, dtype=torch.float32).T  # (2,2)
            rot = pts @ R
            rot[:, 0] += cx
            rot[:, 1] += cy
            return rot  # (4,2)

        def _poly_area(poly):
            x = poly[:, 0]; y = poly[:, 1]
            return 0.5 * torch.abs(torch.sum(x * torch.roll(y, -1)) - torch.sum(y * torch.roll(x, -1)))

        def _poly_clip(subject, clipper):
            # Sutherland–Hodgman polygon clipping (convex assumption).
            def edge_clip(s_poly, p1, p2):
                out = []
                for i in range(len(s_poly)):
                    A = s_poly[i - 1]; Bp = s_poly[i]
                    def inside(P):
                        return (p2[0]-p1[0])*(P[1]-p1[1]) - (p2[1]-p1[1])*(P[0]-p1[0]) >= 0
                    def intersect():
                        x1,y1 = A; x2,y2 = Bp; x3,y3 = p1; x4,y4 = p2
                        den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                        if den == 0:
                            return Bp
                        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / den
                        return torch.tensor([x1 + t*(x2-x1), y1 + t*(y2-y1)], device=device, dtype=torch.float32)
                    Ain, Bin = inside(A), inside(Bp)
                    if Ain and Bin:
                        out.append(Bp)
                    elif Ain and (not Bin):
                        out.append(intersect())
                    elif (not Ain) and Bin:
                        out.append(intersect()); out.append(Bp)
                return out

            output = [p for p in subject]
            for i in range(len(clipper)):
                p1 = clipper[i - 1]; p2 = clipper[i]
                output = edge_clip(output, p1, p2)
                if not output:
                    break
            if not output:
                return None
            return torch.stack(output, dim=0)

        def _polygon_iou(poly1, poly2):
            inter = _poly_clip(poly1, poly2)
            if inter is None or inter.shape[0] < 3:
                return torch.tensor(0.0, device=device)
            ai = _poly_area(inter)
            a1 = _poly_area(poly1)
            a2 = _poly_area(poly2)
            union = a1 + a2 - ai
            return ai / (union + 1e-12)

        def _greedy_nms_polys(polys, scores, iou_thr=0.5, max_det=100):
            order = torch.argsort(scores, descending=True)
            keep = []
            for idx in order:
                idx = idx.item()
                good = True
                for j in keep:
                    if _polygon_iou(polys[idx], polys[j]) > iou_thr:
                        good = False; break
                if good:
                    keep.append(idx)
                if len(keep) >= max_det:
                    break
            return torch.tensor(keep, device=device, dtype=torch.long)

        per_img_boxes = [[] for _ in range(B)]
        per_img_scores = [[] for _ in range(B)]
        per_img_cls = [[] for _ in range(B)]

        for p in pyramids:
            if (not torch.is_tensor(p)) or p.dim() != 4 or p.size(1) < 7:
                continue
            Bp, C, Hf, Wf = p.shape
            if Bp != B:
                raise AssertionError("Batch mismatch between pyramids and imgs")

            stride_x = W / float(Wf)
            stride_y = H / float(Hf)
            stride = 0.5 * (stride_x + stride_y)

            tx = p[:, 0:1]  # (B,1,Hf,Wf)
            ty = p[:, 1:2]
            tw = p[:, 2:3]
            th = p[:, 3:4]
            s = p[:, 4:5]  # sinθ
            c = p[:, 5:6]  # cosθ
            tobj = p[:, 6:7]  # obj
            cls_logits = p[:, 7:] if C > 7 else None

            # grids
            gy = torch.arange(Hf, device=device).view(1, 1, Hf, 1).expand(B, 1, Hf, Wf)
            gx = torch.arange(Wf, device=device).view(1, 1, 1, Wf).expand(B, 1, Hf, Wf)

            # decode to pixels
            cx = (gx + tx.sigmoid()) * stride_x
            cy = (gy + ty.sigmoid()) * stride_y
            w = tw.exp() * stride
            h = th.exp() * stride
            ang = torch.atan2(s, c).squeeze(1)  # (B,Hf,Wf)
            obj = tobj.sigmoid().squeeze(1)     # (B,Hf,Wf)

            if cls_logits is not None and cls_logits.numel():
                cls_prob = cls_logits.sigmoid()           # (B,nc,Hf,Wf)
                cls_score, cls_idx = cls_prob.max(dim=1)  # (B,Hf,Wf)
                score = obj * cls_score
            else:
                score = obj
                cls_idx = None

            # flatten
            cx = cx.reshape(B, -1)
            cy = cy.reshape(B, -1)
            w  = w.reshape(B, -1)
            h  = h.reshape(B, -1)
            ang = ang.reshape(B, -1)
            score = score.reshape(B, -1)
            if cls_idx is not None:
                cls_idx = cls_idx.reshape(B, -1)

            # threshold + pre-topk and collect
            for b in range(B):
                sb = score[b]
                kb = sb >= float(score_thr)
                if kb.any():
                    if pre_topk and kb.sum().item() > pre_topk:
                        top_idx = torch.topk(sb[kb], k=pre_topk, largest=True).indices
                        kb = kb.nonzero(as_tuple=False).squeeze(1)[top_idx]

                    bx = torch.stack([cx[b][kb], cy[b][kb], w[b][kb], h[b][kb], ang[b][kb]], dim=-1)  # radians
                    sc = sb[kb]
                    cl = (cls_idx[b][kb] if cls_idx is not None else None)
                else:
                    bx = imgs.new_zeros((0, 5))
                    sc = imgs.new_zeros((0,))
                    cl = None
                per_img_boxes[b].append(bx)
                per_img_scores[b].append(sc)
                per_img_cls[b].append(cl)

        # concat per-image and apply poly-NMS & cap
        out = []
        for b in range(B):
            if not per_img_boxes[b]:
                out.append({"boxes": imgs.new_zeros((0, 5)), "scores": imgs.new_zeros((0,)), "cls": None,
                            "obb": imgs.new_zeros((0, 5)), "labels": imgs.new_zeros((0,), dtype=torch.long), "polygons": []})
                continue

            bx = torch.cat(per_img_boxes[b], dim=0) if len(per_img_boxes[b]) > 1 else per_img_boxes[b][0]
            sc = torch.cat(per_img_scores[b], dim=0) if len(per_img_scores[b]) > 1 else per_img_scores[b][0]
            cl = None
            if per_img_cls[b] and per_img_cls[b][0] is not None:
                cl = torch.cat(per_img_cls[b], dim=0) if len(per_img_cls[b]) > 1 else per_img_cls[b][0]

            # Build polygons and deg version
            polys = []
            ang_deg = bx[:, 4] * (180.0 / math.pi)
            for i in range(bx.shape[0]):
                polys.append(_obb_corners(bx[i, 0], bx[i, 1], bx[i, 2], bx[i, 3], bx[i, 4]))

            if use_nms and bx.shape[0] > 1:
                keep_idx = _greedy_nms_polys(polys, sc, iou_thr=nms_iou_thr, max_det=max_det)
                bx = bx[keep_idx]
                sc = sc[keep_idx]
                ang_deg = ang_deg[keep_idx]
                polys = [polys[i.item()] for i in keep_idx]
                if cl is not None:
                    cl = cl[keep_idx]
            else:
                if bx.shape[0] > max_det:
                    top = torch.topk(sc, k=max_det, largest=True).indices
                    bx, sc, ang_deg = bx[top], sc[top], ang_deg[top]
                    polys = [polys[i.item()] for i in top]
                    if cl is not None:
                        cl = cl[top]

            # Extended outputs
            obb_deg = bx.clone()
            obb_deg[:, 4] = ang_deg
            labels = (cl.long() if cl is not None else torch.zeros_like(sc, dtype=torch.long))

            out.append({"boxes": bx, "scores": sc, "cls": cl,
                        "obb": obb_deg, "labels": labels, "polygons": polys})
        return out

    @torch.inference_mode()
    def kpt_from_obbs(self,
                      feats,
                      pos_meta,
                      chunk: int = 128,
                      **kwargs):
        """
        Normalize pos_meta to the per-image OBB list that RotatedROIPool expects.
        Accepts:
          • list of dicts {'b'|'bix', 'obb'| 'obb_abs'|'obb_norm'}
          • list/tuple of length B where each item is (Ni,5) Tensor or None
          • flat list of Tensors (various shapes)
          • single Tensor (N,5) or (N,6) (first col may be batch index)
        Returns:
          (kpred, metas)
        """
        feat = self._select_kpt_feat(feats)  # robustly pick one map
        device, dtype = feat.device, feat.dtype
        B, C, Hf, Wf = feat.shape

        def to_tensor(x):
            if torch.is_tensor(x):
                return x.to(device=device, dtype=dtype)
            return torch.tensor(x, device=device, dtype=dtype)

        # --- Build obb_list: list length B, each item is (Ni,5) Tensor or None ---
        obb_list = [None] * B

        if isinstance(pos_meta, (list, tuple)):
            # Case 1: evaluator-style per-image list
            if len(pos_meta) == B and all((x is None) or (torch.is_tensor(x) and x.ndim == 2 and x.shape[1] >= 5)
                                          for x in pos_meta):
                obb_list = [(to_tensor(x) if x is not None else None) for x in pos_meta]

            else:
                # Case 2: training/other — list of dicts OR flat list of tensors
                grouped = [[] for _ in range(B)]
                for m in pos_meta:
                    if isinstance(m, dict):
                        b = int(m.get('b', m.get('bix', 0)))
                        obb = m.get('obb') or m.get('obb_abs') or m.get('obb_norm')
                        if obb is None:
                            continue
                        grouped[b].append(to_tensor(obb).reshape(-1, 5))
                    elif torch.is_tensor(m):
                        t = to_tensor(m)
                        if t.ndim == 2 and t.shape[1] >= 6:
                            bix = t[:, 0].long().clamp_(0, B - 1)
                            obb = t[:, 1:6]
                            for b in range(B):
                                sel = bix == b
                                if sel.any():
                                    grouped[b].append(obb[sel])
                        elif t.ndim == 2 and t.shape[1] >= 5:
                            # no batch info; assign to image 0
                            grouped[0].append(t[:, :5])
                        elif t.ndim == 1 and t.numel() >= 5:
                            grouped[0].append(t[:5].unsqueeze(0))
                        # else: ignore malformed
                    # else: ignore other element types

                for b in range(B):
                    if len(grouped[b]):
                        obb_list[b] = torch.cat(grouped[b], dim=0)
                    else:
                        obb_list[b] = None

        elif torch.is_tensor(pos_meta):
            # Case 3: single tensor
            t = to_tensor(pos_meta)
            if t.ndim == 2 and t.shape[1] >= 6:
                bix = t[:, 0].long().clamp_(0, B - 1)
                obb = t[:, 1:6]
                for b in range(B):
                    sel = bix == b
                    if sel.any():
                        obb_list[b] = obb[sel]
            elif t.ndim == 2 and t.shape[1] >= 5:
                obb_list[0] = t[:, :5]
            else:
                raise ValueError("pos_meta tensor must be (N,5) or (N,6).")
        else:
            raise TypeError(f"Unsupported pos_meta type: {type(pos_meta)}")

        # --- Optional evaluator knobs ---
        scores_list = kwargs.get("scores_list", None)
        topk = kwargs.get("topk", None)
        score_thresh = kwargs.get("score_thresh", None)

        # --- ROI crop (your RotatedROIPool.forward signature already matches this) ---
        crops, metas = self.roi(
            feat,
            obb_list=obb_list,
            scores_list=scores_list,
            topk=topk,
            chunk=chunk,
            score_thresh=score_thresh,
        )

        if crops.numel() == 0:
            return crops, metas

        kpred = self.kpt_head(crops)
        return kpred, metas