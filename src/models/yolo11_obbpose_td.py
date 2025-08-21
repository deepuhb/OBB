# src/models/yolo11_obbpose_td.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn

from src.models.backbones.cspnext11 import Backbone as CSPBackbone, conv_bn_act
from src.models.necks.pan_fpn import PANFPN
from src.models.heads.obbpose_head import OBBPoseHead
from src.models.layers.rot_roi import RotatedROIPool


# ---------------------- Lazy neck wrapper ----------------------
class LazyPANFPN(nn.Module):
    """Wraps PANFPN and instantiates with proper channels on first forward."""
    def __init__(self, neck: Optional[nn.Module] = None):
        super().__init__()
        self.inner = neck  # may be None until we see channels
        self._ch = None

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor):
        if self.inner is None:
            ch = (p3.shape[1], p4.shape[1], p5.shape[1])
            self.inner = PANFPN(ch=ch)
            self._ch = ch
        return self.inner(p3, p4, p5)

    @property
    def ch(self):
        return self._ch if self._ch is not None else getattr(self.inner, "_ch", None)


# ---------------------- Top-down keypoint head ----------------------
class KptTDHead(nn.Module):
    """Predicts (u,v) in [0,1] from rotated crops (C,S,S)."""
    def __init__(self, in_ch: int = 256):
        super().__init__()
        b = 64
        self.net = nn.Sequential(
            conv_bn_act(in_ch, b, 3, 1), nn.MaxPool2d(2),
            conv_bn_act(b, b*2, 3, 1),   nn.MaxPool2d(2),
            conv_bn_act(b*2, b*4, 3, 1), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(b*4, 64), nn.SiLU(),
            nn.Linear(64, 2), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------- Model ----------------------
class YOLO11_OBBPOSE_TD(nn.Module):
    """
    YOLO11-style OBB + top-down keypoint model.
    - Backbone: CSPNext-11 returning (p3=/8, p4=/16, p5=/32)
    - Neck: PAN-FPN returning (n3=/8, d4=/16, d5=/32)
    - Head: OBBPoseHead -> detection maps (7+nc) and kpt maps (3) per level
    - TD keypoint: RotatedROIPool on P3 (/8) + small head -> (u,v) in [0,1]
    """
    def __init__(
        self,
        num_classes: int = 1,
        width: float = 0.5,
        depth: float = 0.33,
        backbone: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        kpt_crop: int = 64,
        kpt_expand: float = 1.25,
    ) -> None:
        super().__init__()
        self.nc = int(num_classes)
        self.width = float(width)
        self.depth = float(depth)

        # modules
        self.backbone = backbone if backbone is not None else CSPBackbone(in_ch=3, width=self.width, depth=self.depth)
        self.neck = LazyPANFPN(neck if neck is not None else None)
        self.head: Optional[OBBPoseHead] = None  # init lazily

        # top-down ROI + head (feat_down=8 for P3 stride)
        self.roi = RotatedROIPool(out_size=kpt_crop, expand=kpt_expand, feat_down=8)
        self.kpt_head: Optional[KptTDHead] = None  # init lazily after first forward

    # -------- internals --------
    def _ensure_heads(self, feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        if self.head is None:
            ch = (feats[0].shape[1], feats[1].shape[1], feats[2].shape[1])
            self.head = OBBPoseHead(ch=ch, num_classes=self.nc)
        if self.kpt_head is None:
            self.kpt_head = KptTDHead(in_ch=feats[0].shape[1])

    # -------- forward --------
    def forward(self, images: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        # backbone → neck
        p3, p4, p5 = self.backbone(images)  # /8, /16, /32
        n3, d4, d5 = self.neck(p3, p4, p5)  # /8, /16, /32
        self._ensure_heads((n3, d4, d5))

        det_maps, kpt_maps = self.head([n3, d4, d5])  # lists of 3 maps
        return {"det": det_maps, "feats": [n3, d4, d5], "kpt_maps": kpt_maps}

    # -------- decode: det maps -> OBB predictions per image --------
    @torch.no_grad()
    def decode_obb_from_pyramids(
        self,
        det_maps: List[torch.Tensor],
        imgs: Optional[torch.Tensor] = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.7,
        multi_label: bool = False,
        agnostic: bool = False,
        max_det: int = 300,
        use_nms: bool = True,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        YOLO11-style decode:
          - scores = obj * sigmoid(cls) for multi-class (else = obj for single-class)
          - multi_label: keep multiple classes per location if above conf_thres
          - preferred rotated NMS: mmcv.ops.nms_rotated; fallback to torchvision.ops.nms_rotated;
            then AABB NMS; finally top-K.
        Returns per-image dicts: boxes(rad), obb(deg), scores, labels
        """
        # --- Resolve rotated NMS backends (prefer MMCV) ---
        mmcv_nms_rot = None
        tv_nms_rot = None
        try:
            from mmcv.ops import nms_rotated as _mmcv_nms_rot
            if callable(_mmcv_nms_rot):
                mmcv_nms_rot = _mmcv_nms_rot
        except Exception:
            mmcv_nms_rot = None
        if mmcv_nms_rot is None:
            try:
                import torchvision
                _tv = getattr(torchvision.ops, "nms_rotated", None)
                if callable(_tv):
                    tv_nms_rot = _tv
            except Exception:
                tv_nms_rot = None

        strides = (8, 16, 32)
        B = det_maps[0].shape[0]
        device = det_maps[0].device
        results: List[Dict[str, torch.Tensor]] = []

        for b in range(B):
            boxes_all, obb_all, scores_all, labels_all = [], [], [], []

            for level, stride in zip(det_maps, strides):
                pred = level[b]  # (C, H, W)
                C, H, W = pred.shape

                tx = pred[0].sigmoid(); ty = pred[1].sigmoid()
                tw = pred[2].exp();     th = pred[3].exp()
                si = pred[4].tanh();    co = pred[5].tanh()
                obj= pred[6].sigmoid()
                cls_logits = pred[7:] if C > 7 else None

                yv, xv = torch.meshgrid(
                    torch.arange(H, device=pred.device),
                    torch.arange(W, device=pred.device),
                    indexing='ij'
                )
                cx = (tx + xv) * stride
                cy = (ty + yv) * stride
                w  = tw * stride
                h  = th * stride
                ang = torch.atan2(si, co)  # radians

                if cls_logits is None or cls_logits.shape[0] == 0:
                    # single-class
                    conf = obj
                    keep = conf > float(conf_thres)
                    if keep.any():
                        boxes_all.append(torch.stack([cx[keep], cy[keep], w[keep], h[keep], ang[keep]], 1))
                        obb_all.append(torch.stack([cx[keep], cy[keep], w[keep], h[keep], torch.rad2deg(ang[keep])], 1))
                        scores_all.append(conf[keep])
                        labels_all.append(torch.zeros_like(conf[keep], dtype=torch.long))
                else:
                    probs = torch.sigmoid(cls_logits)  # (nc, H, W)
                    if multi_label:
                        scores_pc = obj.unsqueeze(0) * probs            # (nc, H, W)
                        cls_idx, yy, xx = (scores_pc > float(conf_thres)).nonzero(as_tuple=True)
                        if cls_idx.numel():
                            s = scores_pc[cls_idx, yy, xx]
                            cxk = cx[yy, xx]; cyk = cy[yy, xx]
                            wk  =  w[yy, xx]; hk  =  h[yy, xx]
                            ak  = ang[yy, xx]
                            boxes_all.append(torch.stack([cxk, cyk, wk, hk, ak], 1))
                            obb_all.append(torch.stack([cxk, cyk, wk, hk, torch.rad2deg(ak)], 1))
                            scores_all.append(s)
                            labels_all.append(cls_idx.long())
                    else:
                        cls_conf, cls_idx = probs.max(dim=0)            # (H, W)
                        conf = obj * cls_conf
                        keep = conf > float(conf_thres)
                        if keep.any():
                            boxes_all.append(torch.stack([cx[keep], cy[keep], w[keep], h[keep], ang[keep]], 1))
                            obb_all.append(torch.stack([cx[keep], cy[keep], w[keep], h[keep], torch.rad2deg(ang[keep])], 1))
                            scores_all.append(conf[keep])
                            labels_all.append(cls_idx[keep].long())

            if len(scores_all) == 0:
                results.append({
                    "boxes": torch.zeros((0,5), device=device),
                    "obb": torch.zeros((0,5), device=device),
                    "scores": torch.zeros((0,), device=device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=device),
                    "polygons": []
                })
                continue

            boxes = torch.cat(boxes_all, 0)        # (N,5) angle in radians
            obb   = torch.cat(obb_all, 0)          # (N,5) angle in degrees
            scores= torch.cat(scores_all, 0)       # (N,)
            labels= torch.cat(labels_all, 0)       # (N,)

            # --- Rotated NMS (prefer MMCV), else TorchVision, else AABB NMS, else top-K ---
            if use_nms and obb.numel():
                # ensure numeric expectations for kernels
                obb = obb.to(torch.float32).contiguous()
                scores = scores.to(torch.float32).contiguous()

                keep_idx = None
                if mmcv_nms_rot is not None:
                    if agnostic:
                        _, keep_idx = mmcv_nms_rot(obb, scores, float(iou_thres))
                    else:
                        chunks = []
                        for c in labels.unique():
                            idx = (labels == c).nonzero(as_tuple=True)[0]
                            if idx.numel():
                                _, k = mmcv_nms_rot(obb[idx], scores[idx], float(iou_thres))
                                chunks.append(idx[k])
                        keep_idx = torch.cat(chunks, 0) if chunks else torch.arange(0, obb.shape[0], device=obb.device)
                elif tv_nms_rot is not None:
                    if agnostic:
                        keep_idx = tv_nms_rot(obb, scores, float(iou_thres))
                    else:
                        chunks = []
                        for c in labels.unique():
                            idx = (labels == c).nonzero(as_tuple=True)[0]
                            if idx.numel():
                                k = tv_nms_rot(obb[idx], scores[idx], float(iou_thres))
                                chunks.append(idx[k])
                        keep_idx = torch.cat(chunks, 0) if chunks else torch.arange(0, obb.shape[0], device=obb.device)
                else:
                    # AABB fallback via torchvision.ops.nms
                    x1 = obb[:, 0] - obb[:, 2] * 0.5
                    y1 = obb[:, 1] - obb[:, 3] * 0.5
                    x2 = obb[:, 0] + obb[:, 2] * 0.5
                    y2 = obb[:, 1] + obb[:, 3] * 0.5
                    aabb = torch.stack([x1, y1, x2, y2], 1).to(torch.float32).contiguous()
                    try:
                        from torchvision.ops import nms as tv_nms
                    except Exception:
                        tv_nms = None
                    if tv_nms is not None:
                        if agnostic:
                            keep_idx = tv_nms(aabb, scores, float(iou_thres))
                        else:
                            chunks = []
                            for c in labels.unique():
                                idx = (labels == c).nonzero(as_tuple=True)[0]
                                if idx.numel():
                                    k = tv_nms(aabb[idx], scores[idx], float(iou_thres))
                                    chunks.append(idx[k])
                            keep_idx = torch.cat(chunks, 0) if chunks else torch.arange(0, aabb.shape[0], device=aabb.device)

                if keep_idx is not None:
                    keep_idx = keep_idx[scores[keep_idx].argsort(descending=True)]
                    if keep_idx.numel() > max_det:
                        keep_idx = keep_idx[:max_det]
                    boxes, obb, scores, labels = boxes[keep_idx], obb[keep_idx], scores[keep_idx], labels[keep_idx]
                else:
                    # Final fallback: top-K
                    if boxes.shape[0] > max_det:
                        topk = torch.topk(scores, k=max_det, sorted=True)
                        keep_idx = topk.indices
                        boxes, obb, scores, labels = boxes[keep_idx], obb[keep_idx], scores[keep_idx], labels[keep_idx]
            else:
                # no NMS requested; cap to max_det
                if boxes.shape[0] > max_det:
                    topk = torch.topk(scores, k=max_det, sorted=True)
                    keep_idx = topk.indices
                    boxes, obb, scores, labels = boxes[keep_idx], obb[keep_idx], scores[keep_idx], labels[keep_idx]

            results.append({
                "boxes": boxes,   # radians
                "obb": obb,       # degrees
                "scores": scores,
                "labels": labels,
                "polygons": []
            })
        return results

    # -------- top-down keypoints from OBBs --------
    @torch.no_grad()
    def kpt_from_obbs(
        self,
        feats: List[torch.Tensor],
        obb_list: List[torch.Tensor],
        scores_list: Optional[List[torch.Tensor]] = None,
        topk: int = 128,
        chunk: int = 128,
        score_thresh: float = 0.25,
    ) -> Tuple[torch.Tensor, List[dict]]:
        # crop from P3 (/8) features
        P3 = feats[0]
        crops, metas = self.roi(
            P3, obb_list,
            scores_list=scores_list, topk=topk, chunk=chunk, score_thresh=score_thresh
        )
        if crops.numel() == 0:
            return P3.new_zeros((0,2)), []

        # ensure kpt head
        if self.kpt_head is None or getattr(self.kpt_head.net[0][0], "in_channels", None) != P3.shape[1]:
            self.kpt_head = KptTDHead(in_ch=P3.shape[1])

        uv = self.kpt_head(crops)  # (N,2) in [0,1]
        return uv, metas
