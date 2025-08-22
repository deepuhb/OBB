
# src/models/necks/pan_fpn.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use the same helper from the backbone
from src.models.backbones.cspnext11 import conv_bn_act


class PANFPN(nn.Module):
    """
    Channel-consistent PAN-FPN:
      - Project P3, P4, P5 to the same width C (default: C = ch[1], i.e., /16 width).
      - Top-down: fuse at /16 then /8.
      - Bottom-up: aggregate back to /16 and /32.
    Returns (N3=/8, D4=/16, D5=/32) each with C channels.
    """
    def __init__(self, ch=(128, 256, 512), out_ch: int | None = None):
        super().__init__()
        c3, c4, c5 = ch
        C = int(out_ch) if out_ch is not None else int(c4)  # unify to /16 width by default

        # lateral projections to unified width C
        self.lat3 = conv_bn_act(c3, C, k=1, s=1)
        self.lat4 = conv_bn_act(c4, C, k=1, s=1)
        self.lat5 = conv_bn_act(c5, C, k=1, s=1)

        # top-down fusions (concat -> 2C, then reduce to C)
        self.fuse4 = conv_bn_act(C * 2, C, k=3, s=1)
        self.fuse3 = conv_bn_act(C * 2, C, k=3, s=1)

        # bottom-up aggregations
        self.down4 = conv_bn_act(C * 2, C, k=3, s=1)  # cat(pool(f3), f4)
        self.down5 = conv_bn_act(C * 2, C, k=3, s=1)  # cat(pool(d4), lat5(p5))

        self._ch_out = (C, C, C)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        # Top-down
        u4 = F.interpolate(self.lat5(p5), scale_factor=2, mode="nearest")   # /32 -> /16
        f4 = self.fuse4(torch.cat([u4, self.lat4(p4)], dim=1))              # /16, C
        u3 = F.interpolate(f4, scale_factor=2, mode="nearest")              # /16 -> /8
        f3 = self.fuse3(torch.cat([u3, self.lat3(p3)], dim=1))              # /8,  C

        # Bottom-up
        d4 = self.down4(torch.cat([F.max_pool2d(f3, 2), f4], dim=1))        # /16, C
        d5 = self.down5(torch.cat([F.max_pool2d(d4, 2), self.lat5(p5)], dim=1))  # /32, C

        return f3, d4, d5

    @property
    def ch_out(self):
        return self._ch_out