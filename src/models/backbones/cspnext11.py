import torch, torch.nn as nn
def conv_bn_act(c1,c2,k=3,s=1,p=None):
    if p is None: p=k//2
    return nn.Sequential(nn.Conv2d(c1,c2,k,s,p,bias=False), nn.BatchNorm2d(c2), nn.SiLU(inplace=True))
class Bottleneck(nn.Module):
    def __init__(self,c1,c2,shortcut=True):
        super().__init__(); c_=int(c2/2)
        self.cv1=conv_bn_act(c1,c_,1,1,0); self.cv2=conv_bn_act(c_,c2,3,1,1); self.add=shortcut and c1==c2
    def forward(self,x):
        y=self.cv2(self.cv1(x)); return x+y if self.add else y
class C3(nn.Module):
    def __init__(self,c1,c2,n=1):
        super().__init__(); c_=int(c2/2)
        self.cv1=conv_bn_act(c1,c_,1,1,0); self.cv2=conv_bn_act(c1,c_,1,1,0); self.m=nn.Sequential(*[Bottleneck(c_,c_) for _ in range(max(1,n))]); self.cv3=conv_bn_act(2*c_,c2,1,1,0)
    def forward(self,x):
        y1=self.m(self.cv1(x)); y2=self.cv2(x); return self.cv3(torch.cat([y1,y2],1))
class Backbone(nn.Module):
    def __init__(self,in_ch=3,width=0.5,depth=0.33):
        super().__init__()
        c1=int(64*width); c2=int(128*width); c3=int(256*width); c4=int(512*width)
        self.stem=conv_bn_act(in_ch,c1,3,2,1)
        self.c2=nn.Sequential(conv_bn_act(c1,c2,3,2,1), C3(c2,c2,int(3*depth)))
        self.c3=nn.Sequential(conv_bn_act(c2,c3,3,2,1), C3(c3,c3,int(6*depth)))
        self.c4=nn.Sequential(conv_bn_act(c3,c4,3,2,1), C3(c4,c4,int(6*depth)))
    def forward(self,x):
        x=self.stem(x); p3=self.c2(x); p4=self.c3(p3); p5=self.c4(p4); return p3,p4,p5
