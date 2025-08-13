# src/models/yolo11_obbpose_td.py
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
    def __init__(self, num_classes=1, width=0.5, depth=0.33, kpt_crop=64, kpt_expand=1.25):
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

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)       # (c2, c3, c5)
        n3, d4, d5 = self.neck(p3, p4, p5)  # (c2, c3, c5)
        det = self.det_head([n3, d4, d5])
        return {"det": det, "feats": [n3, d4, d5]}

    @torch.no_grad()
    def kpt_from_obbs(self, feats, obb_list):
        p3_like = feats[0]  # n3
        crops, metas = self.roi(p3_like, obb_list)
        if crops.numel() == 0:
            return torch.empty(0, 2, device=p3_like.device), []
        uv = self.kpt_head(crops)
        return uv, metas
