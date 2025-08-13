# src/models/necks/pan_fpn.py
import torch
import torch.nn as nn
from ..backbones.cspnext11 import conv_bn_act

class PANFPN(nn.Module):
    """
    YOLO-style PAN-FPN
    Inputs:  p3 (/4), p4 (/8), p5 (/16)
    Outputs: n3 (/4), d4 (/8), d5 (/16)
    ch: (C3, C4, C5) channel sizes that match the backbone's outputs.
    """
    def __init__(self, ch=(128, 256, 512)):
        super().__init__()
        c3, c4, c5 = ch

        # top-down
        self.ups4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.lat4 = conv_bn_act(c5, c4, 1, 1, 0)        # make p5 -> c4 channels
        self.fuse4 = conv_bn_act(c4 + c4, c4, 3, 1, 1)  # concat (upsampled p5, p4)

        self.ups3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.lat3 = conv_bn_act(c4, c3, 1, 1, 0)        # make n4 -> c3 channels
        self.fuse3 = conv_bn_act(c3 + c3, c3, 3, 1, 1)  # concat (upsampled n4, p3)

        # bottom-up (PAN)
        self.down4  = conv_bn_act(c3, c4, 3, 2, 1)      # n3 -> /8
        self.fuse4d = conv_bn_act(c4 + c4, c4, 3, 1, 1) # concat (down n3, n4)

        self.down5  = conv_bn_act(c4, c5, 3, 2, 1)      # d4 -> /16
        self.fuse5d = conv_bn_act(c5 + c5, c5, 3, 1, 1) # concat (down d4, p5)

    def forward(self, p3, p4, p5):
        # top-down
        u4 = self.ups4(self.lat4(p5))                   # upsample to p4 size
        n4 = self.fuse4(torch.cat([u4, p4], dim=1))     # /8, c4

        u3 = self.ups3(self.lat3(n4))                   # upsample to p3 size
        n3 = self.fuse3(torch.cat([u3, p3], dim=1))     # /4, c3

        # bottom-up
        d4 = self.fuse4d(torch.cat([self.down4(n3), n4], dim=1))  # /8, c4
        d5 = self.fuse5d(torch.cat([self.down5(d4), p5], dim=1))  # /16, c5

        return n3, d4, d5
