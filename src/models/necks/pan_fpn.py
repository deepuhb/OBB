
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.backbones.cspnext11 import conv_bn_act

class PANFPN(nn.Module):
    """
    YOLO-style PAN-FPN
    Inputs:  p3 (/8), p4 (/16), p5 (/32)
    Outputs: n3 (/8), d4 (/16), d5 (/32)
    ch: (C3, C4, C5) channel sizes that match the backbone's outputs.
    """
    def __init__(self, ch=(128, 256, 512)):
        super().__init__()
        c3, c4, c5 = ch

        # lateral reduce 1x1
        self.lat5 = conv_bn_act(c5, c4, 1, 1, 0)
        self.lat4 = conv_bn_act(c4, c3, 1, 1, 0)

        # top-down fusions
        self.fuse4 = conv_bn_act(c4 + c4, c4, 3, 1, 1)  # up(P5:/32 -> /16) + p4
        self.fuse3 = conv_bn_act(c3 + c3, c3, 3, 1, 1)  # up(n4:/16 -> /8) + p3

        # bottom-up PAN
        self.down4 = conv_bn_act(c3, c4, 3, 2, 1)       # n3 /8 -> /16
        self.fuse4d = conv_bn_act(c4 + c4, c4, 3, 1, 1) # down(n3)+n4
        self.down5 = conv_bn_act(c4, c5, 3, 2, 1)       # d4 /16 -> /32
        self.fuse5d = conv_bn_act(c5 + c5, c5, 3, 1, 1) # down(d4) + p5

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        # top-down
        u4 = F.interpolate(self.lat5(p5), scale_factor=2, mode="nearest")  # /16 -> /8 (c4)
        n4 = self.fuse4(torch.cat([u4, p4], dim=1))                        # /8  (c4)

        u3 = F.interpolate(self.lat4(n4), scale_factor=2, mode="nearest")  # /8 -> /4 (c3)
        n3 = self.fuse3(torch.cat([u3, p3], dim=1))                        # /4  (c3)

        # bottom-up
        d4 = self.fuse4d(torch.cat([self.down4(n3), n4], dim=1))           # /8  (c4)
        d5 = self.fuse5d(torch.cat([self.down5(d4), p5], dim=1))           # /16 (c5)

        return n3, d4, d5