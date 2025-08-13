import torch, torch.nn as nn
from ..backbones.cspnext11 import conv_bn_act
class OBBPoseHead(nn.Module):
    def __init__(self,ch=(128,256,512), num_classes=1):
        super().__init__(); cout = 7 + (num_classes if num_classes>1 else 0)
        self.det3=nn.Sequential(conv_bn_act(ch[0],ch[0],3,1,1), nn.Conv2d(ch[0],cout,1,1,0))
        self.det4=nn.Sequential(conv_bn_act(ch[1],ch[1],3,1,1), nn.Conv2d(ch[1],cout,1,1,0))
        self.det5=nn.Sequential(conv_bn_act(ch[2],ch[2],3,1,1), nn.Conv2d(ch[2],cout,1,1,0))
        self.kp3=nn.Sequential(conv_bn_act(ch[0],ch[0],3,1,1), nn.Conv2d(ch[0],3,1,1,0))
        self.kp4=nn.Sequential(conv_bn_act(ch[1],ch[1],3,1,1), nn.Conv2d(ch[1],3,1,1,0))
        self.kp5=nn.Sequential(conv_bn_act(ch[2],ch[2],3,1,1), nn.Conv2d(ch[2],3,1,1,0))
    def forward(self,feats):
        n3,d4,d5=feats; det=[self.det3(n3), self.det4(d4), self.det5(d5)]; kp=[self.kp3(n3), self.kp4(d4), self.kp5(d5)]; return det,kp
