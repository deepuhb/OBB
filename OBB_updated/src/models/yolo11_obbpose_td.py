"""
Simplified YOLO‑like model for OBB + keypoint detection.

This module defines a minimal backbone, neck and detection head for
oriented bounding box (OBB) detection with a single keypoint.  The
backbone downsamples the input image and produces three feature maps
corresponding to strides 8, 16 and 32.  The neck is a lightweight
feature pyramid that refines and upsamples these features.  The head
is the ``OBBPoseHead`` defined in ``src/models/heads/obbpose_head.py``.

The design is intentionally simple; it provides a template into which
more sophisticated backbones (e.g. CSPDarknet) or necks (e.g. PANet)
can be inserted.  It exposes a ``reg_max`` argument to match the
distributional bin count used in the loss.
"""

from __future__ import annotations

from typing import Tuple, List
import torch
import torch.nn as nn

from .heads.obbpose_head import OBBPoseHead

import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Convenience module: Conv → BatchNorm → SiLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SimpleBackbone(nn.Module):
    """A minimal backbone that produces three feature maps at different resolutions."""

    def __init__(self, in_ch: int = 3, base_ch: int = 32) -> None:
        super().__init__()
        # stage 1: stride 2 and 4 (P3)
        self.stage1 = nn.Sequential(
            ConvBNAct(in_ch, base_ch, k=3, s=2, p=1),
            ConvBNAct(base_ch, base_ch * 2, k=3, s=2, p=1),
            ConvBNAct(base_ch * 2, base_ch * 2, k=3, s=1, p=1),
        )
        # stage 2: stride 8 (P4)
        self.stage2 = nn.Sequential(
            ConvBNAct(base_ch * 2, base_ch * 4, k=3, s=2, p=1),
            ConvBNAct(base_ch * 4, base_ch * 4, k=3, s=1, p=1),
        )
        # stage 3: stride 16 (P5)
        self.stage3 = nn.Sequential(
            ConvBNAct(base_ch * 4, base_ch * 8, k=3, s=2, p=1),
            ConvBNAct(base_ch * 8, base_ch * 8, k=3, s=1, p=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stage1(x)
        p3 = x
        x = self.stage2(x)
        p4 = x
        x = self.stage3(x)
        p5 = x
        return p3, p4, p5


class SimpleFPN(nn.Module):
    """A lightweight feature pyramid that upsamples and merges features."""

    def __init__(self, channels: Tuple[int, int, int]) -> None:
        super().__init__()
        c3, c4, c5 = channels
        # lateral 1×1 convs
        self.lateral5 = nn.Conv2d(c5, c4, 1, 1, 0)
        self.lateral4 = nn.Conv2d(c4, c3, 1, 1, 0)
        # output convs
        self.output3 = ConvBNAct(c3, c3, k=3, s=1, p=1)
        self.output4 = ConvBNAct(c4, c4, k=3, s=1, p=1)
        self.output5 = ConvBNAct(c5, c5, k=3, s=1, p=1)

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # top‑down pathway
        p5_up = F.interpolate(self.lateral5(p5), size=p4.shape[-2:], mode="nearest")
        p4 = p4 + p5_up
        p4_up = F.interpolate(self.lateral4(p4), size=p3.shape[-2:], mode="nearest")
        p3 = p3 + p4_up
        # apply output convs
        p3 = self.output3(p3)
        p4 = self.output4(p4)
        p5 = self.output5(p5)
        return p3, p4, p5


class YOLO11OBBPOSETD(nn.Module):
    """A simplified YOLO‑style model for oriented bounding boxes and keypoints."""

    def __init__(self, num_classes: int, reg_max: int = 8, base_ch: int = 32) -> None:
        super().__init__()
        self.backbone = SimpleBackbone(in_ch=3, base_ch=base_ch)
        # determine channels of P3,P4,P5 from backbone
        c3 = base_ch * 2  # stage1 output
        c4 = base_ch * 4  # stage2 output
        c5 = base_ch * 8  # stage3 output
        self.neck = SimpleFPN(channels=(c3, c4, c5))
        self.head = OBBPoseHead(ch=(c3, c4, c5), num_classes=num_classes, reg_max=reg_max)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        det_maps, kpt_maps = self.head([p3, p4, p5])
        return det_maps, kpt_maps