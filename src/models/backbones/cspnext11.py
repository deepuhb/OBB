
import torch
import torch.nn as nn

def conv_bn_act(c1: int, c2: int, k: int = 3, s: int = 1, p: int | None = None) -> nn.Module:
    if p is None:
        p = k // 2
    return nn.Sequential(
        nn.Conv2d(c1, c2, k, s, p, bias=False),
        nn.BatchNorm2d(c2),
        nn.SiLU(inplace=True),
    )

class Bottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True):
        super().__init__()
        c_ = int(c2 // 2)
        self.cv1 = conv_bn_act(c1, c_, 1, 1, 0)
        self.cv2 = conv_bn_act(c_, c2, 3, 1, 1)
        self.add = shortcut and (c1 == c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y

class C3(nn.Module):
    """CSP-like block: split -> bottlenecks -> concat -> fuse."""
    def __init__(self, c1: int, c2: int, n: int = 1):
        super().__init__()
        c_ = int(c2 // 2)
        self.cv1 = conv_bn_act(c1, c_, 1, 1, 0)
        self.cv2 = conv_bn_act(c1, c_, 1, 1, 0)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut=True) for _ in range(max(1, int(n)))])
        self.cv3 = conv_bn_act(2 * c_, c2, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))

class CSPBackbone(nn.Module):
    """
    Lightweight CSP-Next style backbone.
    Returns three feature maps:
      p3: /4  (C3)
      p4: /8  (C4)
      p5: /16 (C5)
    """
    def __init__(self, in_ch: int = 3, width: float = 0.5, depth: float = 0.33):
        super().__init__()
        c1 = int(64 * width)
        c2 = int(128 * width)
        c3 = int(256 * width)
        c4 = int(512 * width)
        c5 = int(1024 * width)

        self.stem = conv_bn_act(in_ch, c1, 3, 2, 1)  # /2
        self.c2 = nn.Sequential(conv_bn_act(c1, c2, 3, 2, 1), C3(c2, c2, int(3 * depth)))  # /4
        self.c3 = nn.Sequential(conv_bn_act(c2, c3, 3, 2, 1), C3(c3, c3, int(6 * depth)))  # /8
        self.c4 = nn.Sequential(conv_bn_act(c3, c4, 3, 2, 1), C3(c4, c4, int(6 * depth)))  # /16
        self.c5 = nn.Sequential(conv_bn_act(c4, c5, 3, 2, 1), C3(c5, c5, int(3 * depth)))  # /32

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c2 = self.c2(x)   # /4
        c3 = self.c3(c2)  # /8
        c4 = self.c4(c3)  # /16
        c5 = self.c5(c4)  # /32
        p3, p4, p5 = c3, c4, c5
        return p3, p4, p5