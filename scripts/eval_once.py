# scripts/eval_once.py
from __future__ import annotations
import os, sys, argparse, json
import math
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from types import SimpleNamespace

BASE_STEM_OUT = 64.0  # reference out-ch of stem for width=1.0

def _safe_torch_load(path, trust=False, quiet=False):
    # Robust loader for PyTorch 2.6+ 'weights_only' default and older checkpoints.
    if trust:
        # Direct unsafe load (no warnings). Use only for checkpoints you trust.
        return torch.load(path, map_location='cpu', weights_only=False)
    try:
        # Best effort: allowlist common NumPy globals used in older checkpoints
        import numpy as np
        try:
            from torch.serialization import add_safe_globals
            allow = [np.core.multiarray.scalar, np.dtype]
            # Some environments need these too; ignore if not present
            for maybe in [getattr(getattr(np, 'dtypes', None), 'Float64DType', None)]:
                if maybe is not None:
                    allow.append(maybe)
            add_safe_globals(allow)
        except Exception:
            pass
        return torch.load(path, map_location='cpu')  # weights_only=True default in PyTorch 2.6+
    except Exception:
        if not quiet:
            print('[EVAL-ONCE] falling back to weights_only=False for ckpt (set --trust_ckpt to hide this).', flush=True)
        return torch.load(path, map_location='cpu', weights_only=False)

def _guess_width_from_ckpt_state(state_dict):
    cand = ['backbone.stem.0.weight', 'backbone.stem.conv.weight', 'backbone.stem.conv.0.weight']
    for k in cand:
        w = state_dict.get(k, None)
        if isinstance(w, torch.Tensor) and w.ndim == 4 and w.shape[0] > 0:
            return float(w.shape[0]) / BASE_STEM_OUT
    return None

def _nearest_width(w):
    choices = [0.25, 0.33, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    return min(choices, key=lambda x: abs(x - w))

def _build_model(num_classes, width):
    from src.models.yolo11_obbpose_td import YOLO11_OBBPOSE_TD
    kw = dict(num_classes=num_classes)
    if width is not None:
        kw['width'] = float(width)
    return YOLO11_OBBPOSE_TD(**kw)

def _load_ckpt_safely(model, state_or_path):
    if isinstance(state_or_path, str):
        ckpt = _safe_torch_load(state_or_path)
        state = ckpt.get('model', ckpt)
    else:
        state = state_or_path
    msd = model.state_dict()
    filtered, mismatched = {}, []
    for k, v in state.items():
        if k in msd and isinstance(v, torch.Tensor) and msd[k].shape == v.shape:
            filtered[k] = v
        else:
            mismatched.append(k)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print('[EVAL] smart-loaded {} tensors; skipped={}; missing={} unexpected={}'.format(len(filtered), len(mismatched), len(missing), len(unexpected)), flush=True)
    if mismatched[:5]:
        print('[EVAL] examples of skipped keys:' + ''.join('\n  - {}'.format(k) for k in mismatched[:5]), flush=True)

def _shape(x):
    if x is None: return None
    return tuple(x.shape) if torch.is_tensor(x) else type(x).__name__

def _uv_to_image_points(uv, metas, feat_down=8):
    # Map normalized uv in [0,1] -> normalized [-1,1] and apply affine from metas (torch/numpy ok).
    out = []
    if uv is None or (hasattr(uv, "numel") and uv.numel() == 0) or not metas:
        return out
    # ensure tensor on CPU for safe scalar access
    uv_cpu = uv.detach().cpu() if torch.is_tensor(uv) else torch.tensor(uv, dtype=torch.float32)
    for i, meta in enumerate(metas):
        M = meta.get('M', None)
        bix = int(meta.get('bix', 0))
        if M is None:
            continue
        # make M a 2x3 torch tensor on CPU
        if not torch.is_tensor(M):
            M_t = torch.as_tensor(M, dtype=torch.float32)
        else:
            M_t = M.detach().float().cpu()
        if M_t.ndim == 3 and M_t.shape[0] == 2:
            # already 2x3? handle accidental extra dim
            M_t = M_t
        elif M_t.ndim != 2 or M_t.shape[:2] != (2,3):
            # try to coerce last dims
            M_t = M_t.view(2,3)
        # uv -> [-1,1]
        u = uv_cpu[i, 0] * 2.0 - 1.0
        v = uv_cpu[i, 1] * 2.0 - 1.0
        xy1 = torch.stack([u, v, torch.tensor(1.0)], dim=0)  # (3,)
        xy_feat = M_t @ xy1  # (2,)
        x = float(xy_feat[0] * float(feat_down))
        y = float(xy_feat[1] * float(feat_down))
        out.append((bix, x, y))
    return out


def main():
    ap = argparse.ArgumentParser('Eval once')
    ap.add_argument('--data_root', type=str, default='datasets')
    ap.add_argument('--img_size', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--classes', type=int, default=1)
    ap.add_argument('--ckpt', type=str, default=None, help='optional checkpoint to load')
    ap.add_argument('--width', type=float, default=None, help='width multiplier; if omitted and --ckpt is set, auto-infer')
    ap.add_argument('--conf_thres', type=float, default=0.25)
    ap.add_argument('--iou_thres', type=float, default=0.7)
    ap.add_argument('--max_det', type=int, default=300)
    ap.add_argument('--limit', type=int, default=8, help='max images to evaluate')
    ap.add_argument('--kpt_crop', type=int, default=64)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--trust_ckpt', action='store_true', help='Skip safe-globals dance; load with weights_only=False silently')
    ap.add_argument('--quiet_ckpt', action='store_true', help='Suppress checkpoint load fallbacks/warnings')
    args = ap.parse_args()

    os.environ.setdefault('EVAL_DEBUG', '1')
    device = torch.device(args.device)

    print('[EVAL-ONCE] building dataloaders...', flush=True)
    from src.data.build import build_dataloaders
    cfg = SimpleNamespace(
        data=SimpleNamespace(root=args.data_root, img_size=args.img_size, names=None),
        train=SimpleNamespace(batch=args.batch, workers=args.workers),
        val=SimpleNamespace(batch=args.batch, workers=args.workers),
        seed=None,
        pin_memory=True,
        mosaic=True,
        mixup=False,
        copy_paste=False,
        hsv=True,
        flipud=0.0,
        fliplr=0.5,
        degrees=0.0,
        translate=0.0,
        scale=0.9,
        shear=0.0,
        perspective=0.0,
        mosaic_prob=1.0,
        mixup_prob=0.0,
        kpt_expand=1.25,
        kpt_crop=args.kpt_crop,
    )
    train_loader, val_loader, _ = build_dataloaders(cfg, args.batch, args.workers, 0, 0, 1)
    print('[EVAL-ONCE] dataloaders ready.', flush=True)

    width = args.width
    ckpt_state = None
    if args.ckpt and width is None:
        print("[EVAL-ONCE] probing checkpoint '{}' for width...".format(args.ckpt), flush=True)
        tmp = _safe_torch_load(args.ckpt, trust=args.trust_ckpt, quiet=args.quiet_ckpt)
        ckpt_state = tmp.get('model', tmp)
        w_guess = _guess_width_from_ckpt_state(ckpt_state)
        if w_guess is not None:
            width = _nearest_width(w_guess)
            print('[EVAL] inferred width from ckpt: ~{}'.format(width), flush=True)

    print('[EVAL-ONCE] building model (width={})...'.format(width), flush=True)
    model = _build_model(num_classes=args.classes, width=width).to(device).eval()

    if args.ckpt:
        print("[EVAL-ONCE] loading checkpoint '{}'...".format(args.ckpt), flush=True)
        if ckpt_state is None:
            ckpt_state = _safe_torch_load(args.ckpt, trust=args.trust_ckpt, quiet=args.quiet_ckpt).get('model', _safe_torch_load(args.ckpt, trust=args.trust_ckpt, quiet=args.quiet_ckpt))
        _load_ckpt_safely(model, ckpt_state)

    from src.engine.evaluator import Evaluator
    ev = Evaluator(cfg=dict(
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        use_nms=True,
    ))

    print('[EVAL-ONCE] grabbing one val batch...', flush=True)
    batch = next(iter(val_loader))
    imgs = batch['image'].to(device).float()
    with torch.no_grad():
        outs = model(imgs)
        if isinstance(outs, dict):
            det_maps, feats = outs.get('det'), outs.get('feats')
        elif isinstance(outs, (list, tuple)):
            det_maps, feats = outs[0], (outs[1] if len(outs) > 1 else None)
        else:
            det_maps, feats = outs, None

        preds_list = model.decode_obb_from_pyramids(
            det_maps, imgs,
            conf_thres=args.conf_thres, iou_thres=args.iou_thres,
            multi_label=False, agnostic=False,
            max_det=args.max_det, use_nms=True,
        )

        # --- Keypoints (top-down) ---
        kpts_pred = None
        if hasattr(model, 'kpt_from_obbs') and feats is not None:
            obb_list = [p.get('obb', p.get('boxes')) for p in preds_list]
            score_list = [p.get('scores') for p in preds_list]
            uv_all, metas = model.kpt_from_obbs(feats, obb_list, scores_list=score_list, topk=128, chunk=128, score_thresh=args.conf_thres)
            feat_down = getattr(getattr(model, 'roi', None), 'feat_down', 8)
            kpts_img = _uv_to_image_points(uv_all, metas, feat_down)  # list of (bix, x, y)
            kpts_pred = kpts_img

    print('[EVAL-ONCE] DETS :', [tuple(d.shape) for d in det_maps], flush=True)
    if isinstance(outs, dict) and outs.get('feats') is not None:
        print('[EVAL-ONCE] FEATS:', [tuple(f.shape) for f in outs['feats']], flush=True)
    b0 = 0
    p0 = preds_list[b0]
    print('[EVAL-ONCE] image0: preds boxes:', _shape(p0.get('boxes')), 'scores:', _shape(p0.get('scores')),
          'labels:', _shape(p0.get('labels')), flush=True)
    if 'bboxes' in batch:
        gt0 = batch['bboxes'][b0]
        print('[EVAL-ONCE] image0: GT boxes:', _shape(gt0), flush=True)
    if 'labels' in batch and batch['labels'][b0] is not None:
        gl0 = batch['labels'][b0]
        print('[EVAL-ONCE] image0: GT labels:', _shape(gl0), flush=True)
    if 'kpts' in batch and batch['kpts'][b0] is not None:
        gk0 = batch['kpts'][b0]
        print('[EVAL-ONCE] image0: GT kpts:', _shape(gk0), flush=True)

    # Print a few predicted keypoints for image 0
    if kpts_pred is not None:
        samp = [(bx, x, y) for (bx, x, y) in kpts_pred if bx == 0][:5]
        print('[EVAL-ONCE] image0: PRED kpts (first 5):', [(round(x,2), round(y,2)) for (_,x,y) in samp], flush=True)

    print('[EVAL-ONCE] running full evaluation (limited)...', flush=True)
    metrics = ev.evaluate(model, val_loader, device=device, max_images=args.limit)
    print('[EVAL-ONCE] metrics:', json.dumps(metrics, indent=2, default=float), flush=True)

if __name__ == '__main__':
    main()