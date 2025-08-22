# src/models/necks/pan_fpn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PANFPN(nn.Module):
    """
    Explicit-contract PAN/FPN neck.
    - Takes backbone feature channels explicitly: ch_in=(c3,c4,c5)
    - Emits three maps (all with ch_out channels): P3_out (/8), P4_out (/16), P5_out (/32)
    """
    def __init__(self, ch_in: tuple[int, int, int], ch_out: int):
        super().__init__()
        assert len(ch_in) == 3, "ch_in must be a 3-tuple (c3,c4,c5)"
        c3, c4, c5 = ch_in

        # lateral 1x1 to unify channels
        self.lat5 = nn.Conv2d(c5, ch_out, 1, 1, 0)
        self.lat4 = nn.Conv2d(c4, ch_out, 1, 1, 0)
        self.lat3 = nn.Conv2d(c3, ch_out, 1, 1, 0)

        # top-down smoothing
        self.smooth4 = nn.Conv2d(ch_out, ch_out, 3, 1, 1)
        self.smooth3 = nn.Conv2d(ch_out, ch_out, 3, 1, 1)

        # bottom-up aggregation
        self.down3 = nn.Conv2d(ch_out, ch_out, 3, 2, 1)         # P3_out -> stride-16
        self.out4  = nn.Conv2d(ch_out * 2, ch_out, 3, 1, 1)     # cat(P3_down, P4_top)
        self.down4 = nn.Conv2d(ch_out, ch_out, 3, 2, 1)         # P4_out -> stride-32
        self.out5  = nn.Conv2d(ch_out * 2, ch_out, 3, 1, 1)     # cat(P4_down, P5_top)

        # expose for the head
        self.ch_out = ch_out

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        # Top-down
        p5_top = self.lat5(p5)
        p4_top = self.lat4(p4) + F.interpolate(p5_top, scale_factor=2, mode="nearest")
        p4_top = self.smooth4(p4_top)

        p3_top = self.lat3(p3) + F.interpolate(p4_top, scale_factor=2, mode="nearest")
        p3_top = self.smooth3(p3_top)

        # Bottom-up
        p3_down = self.down3(p3_top)
        p4_out  = self.out4(torch.cat([p3_down, p4_top], dim=1))
        p4_down = self.down4(p4_out)
        p5_out  = self.out5(torch.cat([p4_down, p5_top], dim=1))

        # Final three maps (all ch_out channels): /8, /16, /32
        return p3_top, p4_out, p5_out
