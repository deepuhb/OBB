# scripts/eval_once.py (deterministic, VAL-only, device-safe)
from __future__ import annotations
import os, sys, argparse, json, random
import numpy as np
import torch
from src.utils.model_schema import print_schema
from src.data.build import build_dataloaders

# Make eval deterministic across runs
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
try:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except Exception:
    pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BASE_STEM_OUT = 64.0  # reference conv-out for width=1.0 in our CSPNext stem


def _safe_torch_load(path, trust=False, quiet=False):
    """Robust loader for PyTorch 2.6+ `weights_only` default with older ckpts."""
    if trust:
        return torch.load(path, map_location='cpu', weights_only=False)
    try:
        # Allowlist common NumPy globals that appear in historical checkpoints
        try:
            from torch.serialization import add_safe_globals
            import numpy as _np
            add_safe_globals([_np.core.multiarray.scalar, _np.dtype])
        except Exception:
            pass
        return torch.load(path, map_location='cpu')  # weights_only=True in PyTorch 2.6
    except Exception:
        if not quiet:
            print('[EVAL-ONCE] torch.load(weights_only=True) failed; falling back to weights_only=False.')
        return torch.load(path, map_location='cpu', weights_only=False)


def _guess_width_from_ckpt_state(state_dict):
    for k in ('backbone.stem.0.weight', 'backbone.stem.conv.weight', 'backbone.stem.conv.0.weight'):
        w = state_dict.get(k, None)
        if isinstance(w, torch.Tensor) and w.ndim == 4 and w.shape[0] > 0:
            return float(w.shape[0]) / BASE_STEM_OUT
    return None


def _nearest_width(w):
    choices = [0.25, 0.33, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    return min(choices, key=lambda x: abs(x - w))


def _build_model(num_classes, width):
    from src.models.yolo11_obbpose_td import YOLO11_OBBPOSE_TD
    return YOLO11_OBBPOSE_TD(num_classes=num_classes, width=float(width))


def _smart_load_and_zero_unmatched(model, state_or_path, quiet=False):
    if isinstance(state_or_path, str):
        ckpt = _safe_torch_load(state_or_path, quiet=quiet)
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
    print('[EVAL] smart-loaded {} tensors; skipped={}; missing={} unexpected={}'.format(
        len(filtered), len(mismatched), len(missing), len(unexpected)), flush=True)
    if mismatched[:5]:
        print('[EVAL] examples of skipped keys:' + ''.join('\n  - {}'.format(k) for k in mismatched[:5]), flush=True)

    # Zero-init any params/buffers NOT loaded so eval is reproducible
    loaded_keys = set(filtered.keys())
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name not in loaded_keys:
                p.zero_()
        for name, b in model.named_buffers():
            if name not in loaded_keys:
                try:
                    b.zero_()
                except Exception:
                    pass


def _uv_to_image_points(uv, metas, feat_down=8):
    """Map normalized uv in [0,1] -> image px using affine M in metas and feature downsample.
    Device-safe: uses the device of M.
    """
    out = []
    if uv is None or (hasattr(uv, 'numel') and uv.numel() == 0) or not metas:
        return out
    for i, meta in enumerate(metas):
        M = meta.get('M', None)
        bix = int(meta.get('bix', meta.get('b', 0)))
        if M is None:
            continue
        M_t = torch.as_tensor(M, dtype=torch.float32, device=M.device if torch.is_tensor(M) else None)
        dev = M_t.device
        # convert uv[i] to same device
        uv_i = uv[i].to(dev)
        # uv [0,1] -> [-1,1]
        u = uv_i[0] * 2.0 - 1.0
        v = uv_i[1] * 2.0 - 1.0
        ones = torch.ones((), device=dev, dtype=torch.float32)
        xy1 = torch.stack([u, v, ones], dim=0)  # (3,)
        M2x3 = M_t.view(2, 3)
        xy_feat = M2x3 @ xy1  # (2,)
        x = float(xy_feat[0] * float(feat_down))
        y = float(xy_feat[1] * float(feat_down))
        out.append((bix, x, y))
    return out


def main():
    ap = argparse.ArgumentParser('Eval once (VAL split)')
    ap.add_argument('--data_root', type=str, default='datasets')
    ap.add_argument('--img_size', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--classes', type=int, default=1)
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--width', type=float, default=None, help='width multiplier; if omitted and --ckpt is set, auto-infer')
    ap.add_argument('--conf_thres', type=float, default=0.05)
    ap.add_argument('--iou_thres', type=float, default=0.5)
    ap.add_argument('--max_det', type=int, default=300)
    ap.add_argument('--limit', type=int, default=20, help='max images to evaluate')
    ap.add_argument('--kpt_crop', type=int, default=64)
    ap.add_argument('--kpt_topk', type=int, default=128)
    ap.add_argument('--kpt_force', action='store_true', help='Ignore scores for kpt; take top-K per batch')
    ap.add_argument('--kpt_min_per_img', type=int, default=1, help='Guarantee at least this many kpt crops per image')
    ap.add_argument('--kpt_fallback_frac', type=float, default=0.5, help='If no boxes, use center OBB of this fraction of min(H,W)')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--trust_ckpt', action='store_true', help='Skip safe-globals dance; load with weights_only=False silently')
    ap.add_argument('--quiet_ckpt', action='store_true', help='Suppress checkpoint load fallbacks/warnings')
    args = ap.parse_args()

    device = torch.device(args.device)

    print('[EVAL-ONCE] building dataloaders...', flush=True)
    from types import SimpleNamespace as NS
    cfg = NS(
        data=NS(root=args.data_root, img_size=args.img_size, names=None),
        train=NS(batch=args.batch, workers=args.workers),
        val=NS(batch=args.batch, workers=args.workers),
        seed=0,
        pin_memory=True,
        mosaic=False, mixup=False, copy_paste=False, hsv=False,
        flipud=0.0, fliplr=0.0, degrees=0.0, translate=0.0, scale=1.0, shear=0.0, perspective=0.0,
        mosaic_prob=0.0, mixup_prob=0.0, kpt_expand=1.25, kpt_crop=args.kpt_crop,
    )

    train_loader, val_loader, _ = build_dataloaders(cfg, args.batch, args.workers, 0, 0, 1)
    print('[EVAL-ONCE] dataloaders ready.', flush=True)
    # Show VAL sample path summary (if dataset exposes)
    val_ds = getattr(val_loader, 'dataset', None)
    if hasattr(val_ds, 'im_files'):
        n = len(getattr(val_ds, 'im_files', []))
        print('[EVAL-ONCE] VAL split ({} files)'.format(n), flush=True)
    else:
        print('[EVAL-ONCE] VAL split (paths not exposed by dataset)', flush=True)

    # Probe width from checkpoint if needed
    width = args.width
    ckpt_state = None
    if args.ckpt and width is None:
        print(f"[EVAL-ONCE] probing checkpoint '{args.ckpt}' for width...", flush=True)
        tmp = _safe_torch_load(args.ckpt, trust=args.trust_ckpt, quiet=args.quiet_ckpt)
        ckpt_state = tmp.get('model', tmp)
        w_guess = _guess_width_from_ckpt_state(ckpt_state)
        if w_guess is not None:
            choices = [0.25, 0.33, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            width = min(choices, key=lambda x: abs(x - w_guess))
            print(f"[EVAL] inferred width from ckpt: ~{width}", flush=True)

    # Build model
    print(f"[EVAL-ONCE] building model (width={width})...", flush=True)
    model = _build_model(num_classes=args.classes, width=width).to(device).eval()
    print_schema(model, tag='Eval-CONSTRUCTED')

    # Load checkpoint (smart-load + zero unmatched)
    print(f"[EVAL-ONCE] loading checkpoint '{args.ckpt}'...", flush=True)
    if ckpt_state is None:
        ckpt_state = _safe_torch_load(args.ckpt, trust=args.trust_ckpt, quiet=args.quiet_ckpt).get('model', _safe_torch_load(args.ckpt, trust=args.trust_ckpt, quiet=args.quiet_ckpt))
    _smart_load_and_zero_unmatched(model, ckpt_state, quiet=args.quiet_ckpt)

    # One VAL batch diagnostics
    print('[EVAL-ONCE] grabbing one val batch...', flush=True)
    batch = next(iter(val_loader))
    imgs = (batch.get('image')).to(device).float()

    with torch.no_grad():
        outs = model(imgs)
        det_maps = outs['det'] if isinstance(outs, dict) else outs[0]
        feats = outs.get('feats') if isinstance(outs, dict) else (outs[1] if isinstance(outs, (list, tuple)) and len(outs) > 1 else None)

        # Decode with thresholds
        try:
            preds = model.decode_obb_from_pyramids(det_maps, imgs,
                                                   conf_thres=args.conf_thres, iou_thres=args.iou_thres,
                                                   max_det=args.max_det, use_nms=True)
        except TypeError:
            preds = model.decode_obb_from_pyramids(det_maps, imgs,
                                                   score_thr=args.conf_thres, iou_thr=args.iou_thres,
                                                   max_det=args.max_det, use_nms=True)

        # Keypoints via top-down head (force at least one crop per image if --kpt_force)
        kpts_img = []
        if feats is not None and hasattr(model, 'kpt_from_obbs'):
            B, _, Himg, Wimg = imgs.shape
            obb_list_raw = [p['obb'] if 'obb' in p else p.get('boxes') for p in preds]
            scores_raw = [p.get('scores') for p in preds]

            obb_list, scores_list = [], []
            for i in range(B):
                o = obb_list_raw[i]
                s = scores_raw[i]
                if o is None or (hasattr(o, 'numel') and o.numel() == 0):
                    o = torch.empty((0,5), device=imgs.device, dtype=torch.float32)
                    s = torch.empty((0,), device=imgs.device, dtype=torch.float32)

                if not args.kpt_force and s is not None and s.numel() > 0:
                    keep = s >= args.conf_thres
                    o2 = o[keep] if keep.any() else o[:0]
                    s2 = s[keep] if keep.any() else s[:0]
                else:
                    o2, s2 = o, s if s is not None else torch.empty((o.shape[0],), device=o.device)

                need = max(0, args.kpt_min_per_img - int(o2.shape[0]))
                if need > 0:
                    if o.shape[0] > 0:
                        if s.shape[0] == 0:
                            s = torch.linspace(1, 0, steps=o.shape[0], device=o.device)
                        order = torch.argsort(s, descending=True)
                        to_add = o[order[:need]]
                        o2 = torch.cat([o2, to_add], dim=0) if o2.numel() else to_add
                        s2 = torch.cat([s2, s[order[:need]]], dim=0) if s2.numel() else s[order[:need]]
                    else:
                        w0 = h0 = float(min(Himg, Wimg)) * float(args.kpt_fallback_frac)
                        cx0 = float(Wimg) * 0.5
                        cy0 = float(Himg) * 0.5
                        synth = torch.tensor([[cx0, cy0, w0, h0, 0.0]], device=imgs.device, dtype=torch.float32).repeat(need,1)
                        o2 = torch.cat([o2, synth], dim=0) if o2.numel() else synth
                        s_synth = torch.ones((need,), device=imgs.device, dtype=torch.float32) * 0.5
                        s2 = torch.cat([s2, s_synth], dim=0) if s2.numel() else s_synth

                obb_list.append(o2)
                scores_list.append(s2)

            uv_all, metas = model.kpt_from_obbs(
                feats, obb_list, scores_list=(None if args.kpt_force else scores_list),
                topk=args.kpt_topk, chunk=128,
                score_thresh=(args.conf_thres if not args.kpt_force else 0.0)
            )
            feat_down = getattr(getattr(model, 'roi', None), 'feat_down', 8)
            print(f'[EVAL-ONCE] kpt_from_obbs: uv_all={tuple(uv_all.shape) if hasattr(uv_all, "shape") else None} metas={len(metas)}', flush=True)
            kpts_img = _uv_to_image_points(uv_all, metas, feat_down=feat_down)

    print('[EVAL-ONCE] DETS :', [tuple(d.shape) for d in det_maps], flush=True)
    if feats is not None:
        print('[EVAL-ONCE] FEATS:', [tuple(f.shape) for f in feats], flush=True)

    p0 = preds[0]
    print('[EVAL-ONCE] image0: preds boxes:', tuple(p0['obb'].shape) if 'obb' in p0 else tuple(p0['boxes'].shape),
          'scores:', tuple(p0['scores'].shape), 'labels:', tuple(p0['labels'].shape), flush=True)

    if 'bboxes' in batch:
        print('[EVAL-ONCE] image0: GT boxes:', tuple(batch['bboxes'][0].shape), flush=True)
    if 'labels' in batch:
        print('[EVAL-ONCE] image0: GT labels:', tuple(batch['labels'][0].shape), flush=True)
    if 'kpts' in batch:
        print('[EVAL-ONCE] image0: GT kpts:', tuple(batch['kpts'][0].shape), flush=True)

    if kpts_img:
        k0 = [(bx, x, y) for (bx, x, y) in kpts_img if bx == 0][:5]
        print('[EVAL-ONCE] image0: PRED kpts (first 5):', [(round(x,2), round(y,2)) for (_,x,y) in k0], flush=True)
    else:
        print('[EVAL-ONCE] image0: PRED kpts (first 5): []', flush=True)

    # Full evaluation on VAL (limited)
    from src.engine.evaluator import Evaluator
    ev = Evaluator(cfg=dict(conf_thres=args.conf_thres, iou_thres=args.iou_thres, max_det=args.max_det, use_nms=True))
    print('[EVAL-ONCE] running full evaluation (limited)...', flush=True)
    metrics = ev.evaluate(model, val_loader, device=device, max_images=args.limit)
    print('[EVAL-ONCE] metrics:', json.dumps(metrics, indent=2, default=float), flush=True)


if __name__ == '__main__':
    main()