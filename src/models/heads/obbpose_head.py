# src/models/heads/obbpose_head.py

import torch
import torch.nn as nn
from typing import List, Tuple
from src.models.backbones.cspnext11 import conv_bn_act

class OBBPoseHead(nn.Module):
    """
    Multi-scale head for OBB detection and bottom-up keypoint hints.

    - det: per-level conv to (7 + nc) channels
           [tx, ty, tw, th, sin, cos, obj, (cls...)]
    - kp:  per-level conv to 3 channels (u, v, score) as heatmap hints (optional use)
    """

    def __init__(self, ch: Tuple[int, int, int], num_classes: int):
        super().__init__()
        c3, c4, c5 = ch
        det_out = 5 + 1 + num_classes  # (cx, cy, w, h, th) + obj + cls
        kpt_out = 3  # your 3 heatmaps / maps

        self.det3 = nn.Sequential(
            nn.Conv2d(c3, c3, 3, 1, 1), nn.BatchNorm2d(c3), nn.SiLU(inplace=True),
            nn.Conv2d(c3, det_out, 1, 1, 0),
        )
        self.det4 = nn.Sequential(
            nn.Conv2d(c4, c4, 3, 1, 1), nn.BatchNorm2d(c4), nn.SiLU(inplace=True),
            nn.Conv2d(c4, det_out, 1, 1, 0),
        )
        self.det5 = nn.Sequential(
            nn.Conv2d(c5, c5, 3, 1, 1), nn.BatchNorm2d(c5), nn.SiLU(inplace=True),
            nn.Conv2d(c5, det_out, 1, 1, 0),
        )

        self.kp3 = nn.Sequential(
            nn.Conv2d(c3, c3, 3, 1, 1), nn.BatchNorm2d(c3), nn.SiLU(inplace=True),
            nn.Conv2d(c3, kpt_out, 1, 1, 0),
        )
        self.kp4 = nn.Sequential(
            nn.Conv2d(c4, c4, 3, 1, 1), nn.BatchNorm2d(c4), nn.SiLU(inplace=True),
            nn.Conv2d(c4, kpt_out, 1, 1, 0),
        )
        self.kp5 = nn.Sequential(
            nn.Conv2d(c5, c5, 3, 1, 1), nn.BatchNorm2d(c5), nn.SiLU(inplace=True),
            nn.Conv2d(c5, kpt_out, 1, 1, 0),
        )

    def forward(self, feats: List[torch.Tensor]):
        n3, d4, d5 = feats
        det = [self.det3(n3), self.det4(d4), self.det5(d5)]
        kpm = [self.kp3(n3), self.kp4(d4), self.kp5(d5)]
        return det, kpm
