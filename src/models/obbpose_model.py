import torch.nn as nn
from .backbones.cspnext11 import Backbone
from .necks.pan_fpn import PANFPN
from .heads.obbpose_head import OBBPoseHead
class OBBPoseModel(nn.Module):
    def __init__(self, num_classes=1, width=0.5, depth=0.33):
        super().__init__()
        self.backbone=Backbone(in_ch=3, width=width, depth=depth)
        ch=(int(128*width), int(256*width), int(512*width))
        self.neck=PANFPN(ch=ch)
        self.head=OBBPoseHead(ch=ch, num_classes=num_classes)
    def forward(self,x):
        feats=self.backbone(x); feats=self.neck(*feats); det,kp=self.head(feats); return {'det':det, 'kpt':kp}
