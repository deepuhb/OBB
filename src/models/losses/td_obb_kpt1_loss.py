# src/models/losses/td_obb_kpt1_loss.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..layers.rot_roi import RotatedROIPool, obb_to_affine

def _sigmoid(x): return x.sigmoid()
def _tanh(x): return x.tanh()

def _bbox_xyxy(cx, cy, w, h):
    return torch.stack([cx-0.5*w, cy-0.5*h, cx+0.5*w, cy+0.5*h], dim=-1)

class TDOBBWKpt1Criterion(nn.Module):
    """
    Top-down criterion:
      - detection losses on OBB head (box, angle, obj, cls)
      - keypoint loss on rotated crops from P3 using *GT OBBs* (teacher forcing)
    """
    def __init__(self, strides=(8,16,32), num_classes=1,
                 lambda_box=7.5, lambda_obj=3.0, lambda_ang=1.0,
                 lambda_cls=1.0,
                 lambda_kpt=2.0, kpt_crop=64, kpt_expand=1.25,
                 kpt_freeze_epochs=0, kpt_warmup_epochs=0, kpt_iou_gate=0.0):
        super().__init__()
        self.strides = strides
        self.nc = num_classes
        self.lambda_box = float(lambda_box)
        self.lambda_obj = float(lambda_obj)
        self.lambda_ang = float(lambda_ang)
        self.lambda_cls = float(lambda_cls)
        self.lambda_kpt = float(lambda_kpt)
        self.kpt_freeze_epochs = int(kpt_freeze_epochs)
        self.kpt_warmup_epochs = int(kpt_warmup_epochs)
        self.kpt_iou_gate = float(kpt_iou_gate)
        self.roi = RotatedROIPool(out_size=int(kpt_crop), expand=float(kpt_expand), feat_down=int(strides[0]))
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _build_targets(self, det_maps, batch):
        """
        Very simple assigner: for each GT, take nearest cell on P3 (stride s0) and build targets.
        You can expand to multi-level 3x3 neighborhood if you want.
        """
        device = det_maps[0].device
        s = self.strides[0]
        dm = det_maps[0]  # (B,C,H,W), use highest resolution for simplicity
        B, C, H, W = dm.shape
        # Create meshgrid
        ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        xs = xs.reshape(1,1,H,W) * s
        ys = ys.reshape(1,1,H,W) * s

        # targets
        tgt = {
            "mask": torch.zeros((B,1,H,W), dtype=torch.bool, device=device),
            "tx": torch.zeros((B,1,H,W), device=device),
            "ty": torch.zeros((B,1,H,W), device=device),
            "tw": torch.zeros((B,1,H,W), device=device),
            "th": torch.zeros((B,1,H,W), device=device),
            "sin": torch.zeros((B,1,H,W), device=device),
            "cos": torch.zeros((B,1,H,W), device=device),
            "obj": torch.zeros((B,1,H,W), device=device),
            "cls": torch.zeros((B,self.nc,H,W), device=device),
        }
        pos_meta = []  # list of (b, gy, gx, cx, cy, w, h, ang)

        for b in range(B):
            if len(batch["boxes"][b]) == 0:
                continue
            boxes = batch["boxes"][b].to(device)
            if "angles" in batch and len(batch["angles"][b]):
                ang = batch["angles"][b].to(device)
            else:
                ang = torch.zeros((boxes.shape[0],), device=device)
            # choose P3 cell nearest to center
            cx = 0.5*(boxes[:,0] + boxes[:,2])
            cy = 0.5*(boxes[:,1] + boxes[:,3])
            w  = (boxes[:,2]-boxes[:,0]).clamp_min(1.0)
            h  = (boxes[:,3]-boxes[:,1]).clamp_min(1.0)

            gx = (cx / s).long().clamp(0, W-1)
            gy = (cy / s).long().clamp(0, H-1)
            for j in range(cx.numel()):
                xg, yg = gx[j].item(), gy[j].item()
                tgt["mask"][b,0,yg,xg] = True
                # offsets normalized to cell: tx,ty in [0,1)
                cx_cell = (cx[j] - xg*s) / s
                cy_cell = (cy[j] - yg*s) / s
                tgt["tx"][b,0,yg,xg] = cx_cell
                tgt["ty"][b,0,yg,xg] = cy_cell
                tgt["tw"][b,0,yg,xg] = torch.log(w[j].clamp_min(1.0)/s)
                tgt["th"][b,0,yg,xg] = torch.log(h[j].clamp_min(1.0)/s)
                tgt["sin"][b,0,yg,xg] = torch.sin(ang[j])
                tgt["cos"][b,0,yg,xg] = torch.cos(ang[j])
                tgt["obj"][b,0,yg,xg] = 1.0
                # single-class? else use batch["labels"][b]
                if "labels" in batch and len(batch["labels"][b]):
                    c = int(batch["labels"][b][j].item())
                    if 0 <= c < self.nc:
                        tgt["cls"][b,c,yg,xg] = 1.0
                pos_meta.append((b, yg, xg, cx[j], cy[j], w[j], h[j], ang[j]))
        return tgt, pos_meta

    def _loss_det(self, det_maps, tgt):
        dm = det_maps[0]  # (B,7+nc,H,W)
        tx, ty, tw, th, si, co, obj = dm[:,0:1], dm[:,1:2], dm[:,2:3], dm[:,3:4], dm[:,4:5], dm[:,5:6], dm[:,6:7]
        cls = dm[:,7:7+self.nc] if self.nc>1 else None
        m = tgt["mask"]

        # box center offsets in (0..1) with sigmoid
        l_tx = F.l1_loss(_sigmoid(tx)[m], tgt["tx"][m], reduction="mean")
        l_ty = F.l1_loss(_sigmoid(ty)[m], tgt["ty"][m], reduction="mean")
        # size in log-space
        l_tw = F.l1_loss(tw[m], tgt["tw"][m], reduction="mean")
        l_th = F.l1_loss(th[m], tgt["th"][m], reduction="mean")
        l_box = l_tx + l_ty + l_tw + l_th

        # angle via sin/cos regression
        l_ang = F.l1_loss(_tanh(si)[m], tgt["sin"][m], reduction="mean") + \
                F.l1_loss(_tanh(co)[m], tgt["cos"][m], reduction="mean")

        # objectness (pos/neg)
        l_obj_pos = self.bce(obj[m], tgt["obj"][m]).mean() if m.any() else torch.zeros((), device=dm.device)
        l_obj_neg = self.bce(obj[~m], tgt["obj"][~m]).mean() if (~m).any() else torch.zeros((), device=dm.device)
        l_obj = 0.5*(l_obj_pos + l_obj_neg)

        # classification (if nc>1)
        if self.nc > 1 and cls is not None:
            l_cls = self.bce(cls[m.expand_as(cls)], tgt["cls"][m.expand_as(cls)]).mean() if m.any() \
                    else torch.zeros((), device=dm.device)
        else:
            l_cls = torch.zeros((), device=dm.device)

        return l_box, l_ang, l_obj, l_cls

    def _uv_from_kpt(self, kpt_xy, M, S):
        """Map image keypoint (x,y) to crop uv in [0,1] using affine M (crop->feat)."""
        # We applied crops on features; but teacher forcing uses GT OBBs in IMAGE pixels.
        # Our M already expects input coords in feature space; here we only use it to map (crop->feat).
        # For uv, invert: [x_feat, y_feat] = M * [uS, vS, 1]; we want [u,v] in [0,1]
        A = M[:,:2]; b = M[:,2]
        Ainv = torch.inverse(A)
        xy = kpt_xy - b
        uvS = Ainv @ xy
        uv = uvS / float(S)
        return uv

    def forward(self, det_maps, feats, batch, model=None, epoch=0):
        """
        det_maps: list of per-level logits (we use level 0 for assign/loss)
        feats:    [P3,P4,P5]
        batch keys (per image list tensors): 'boxes','angles','labels','kpts'
        model: YOLO11_OBBPOSE_TD instance (for kpt_head), required for TD path
        """
        device = det_maps[0].device
        # ---- detection targets and loss
        tgt, pos_meta = self._build_targets(det_maps, batch)
        l_box, l_ang, l_obj, l_cls = self._loss_det(det_maps, tgt)

        # ---- keypoint TD loss using GT OBBs, crops from P3 features
        # Build per-image OBB list in IMAGE px
        obb_list = []
        kpt_list = []
        for i in range(len(batch["boxes"])):
            if len(batch["boxes"][i]) == 0:
                obb_list.append(torch.empty(0,5, device=device))
                kpt_list.append(torch.empty(0,2, device=device))
                continue
            bxs = batch["boxes"][i].to(device)
            ang = batch["angles"][i].to(device) if "angles" in batch and len(batch["angles"][i]) else torch.zeros((bxs.shape[0],), device=device)
            cx = 0.5*(bxs[:,0] + bxs[:,2]); cy = 0.5*(bxs[:,1] + bxs[:,3])
            w  = (bxs[:,2]-bxs[:,0]).clamp_min(1.0); h=(bxs[:,3]-bxs[:,1]).clamp_min(1.0)
            obb = torch.stack([cx,cy,w,h, ang*(180.0/math.pi)], dim=-1)
            obb_list.append(obb)
            kpt_list.append(batch["kpts"][i].to(device) if len(batch["kpts"][i]) else torch.empty(0,2, device=device))

        p3 = feats[0]
        crops, metas = self.roi(p3, obb_list)  # crops from P3
        if crops.numel() == 0 or model is None:
            l_kpt = torch.zeros((), device=device)
        else:
            uv = model.kpt_head(crops)  # (N,2) in [0,1]
            # Build uv GT list aligned to metas
            S = self.roi.S
            uv_gts = []
            keep_mask = []
            idx_m = 0
            for bix, obb in enumerate(obb_list):
                for j in range(obb.shape[0]):
                    kp = kpt_list[bix][j]
                    M = metas[idx_m]["M"]  # 2x3 mapping crop->feat
                    # We created M in feature coordinates; but our kp is in IMAGE coords.
                    # The affine was constructed from IMAGE obb scaled by 1/stride inside roi() -> handled there.
                    # Here, we only need uv target in crop space: invert linear part as in _uv_from_kpt.
                    A = M[:,:2]; b = M[:,2]
                    Ainv = torch.inverse(A)
                    uvS = Ainv @ (kp - b)
                    uv_gt = uvS / float(S)
                    uv_gts.append(uv_gt)
                    keep_mask.append(True)
                    idx_m += 1
            if len(uv_gts):
                uv_gt = torch.stack(uv_gts, dim=0).clamp(0.0, 1.0)
                l_kpt = F.l1_loss(uv, uv_gt, reduction="mean")
            else:
                l_kpt = torch.zeros((), device=device)

        # curriculum: optionally freeze/warmup kpt loss
        if epoch < self.kpt_freeze_epochs:
            l_kpt_eff = torch.zeros_like(l_kpt)
        elif self.kpt_warmup_epochs > 0:
            t = min(1.0, (epoch - self.kpt_freeze_epochs) / float(self.kpt_warmup_epochs))
            l_kpt_eff = l_kpt * t
        else:
            l_kpt_eff = l_kpt

        loss = self.lambda_box*l_box + self.lambda_ang*l_ang + self.lambda_obj*l_obj + self.lambda_cls*l_cls + self.lambda_kpt*l_kpt_eff
        logs = dict(
            loss=loss.item(),
            loss_box=(self.lambda_box*l_box).item(),
            loss_ang=(self.lambda_ang*l_ang).item(),
            loss_obj=(self.lambda_obj*l_obj).item(),
            loss_cls=(self.lambda_cls*l_cls).item() if self.nc>1 else 0.0,
            loss_kpt=(self.lambda_kpt*l_kpt_eff).item(),
            num_pos=int(tgt["mask"].sum().item())
        )
        return loss, logs
