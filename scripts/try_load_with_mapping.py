#!/usr/bin/env python
import argparse, os, sys
import torch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def _safe_torch_load(path):
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([np._core.multiarray.scalar, np.dtype])
    except Exception:
        try:
            from torch.serialization import add_safe_globals
            add_safe_globals([np.core.multiarray.scalar, np.dtype])  # fallback alias
        except Exception:
            pass
    try:
        return torch.load(path, map_location='cpu')  # PyTorch 2.6 default weights_only=True
    except Exception:
        return torch.load(path, map_location='cpu', weights_only=False)

def apply_mappings(state_dict, maps):
    rules = []
    for m in maps or []:
        if '->' in m:
            a,b = m.split('->',1)
            rules.append((a.strip(), b.strip()))
    if not rules:
        return state_dict, []
    out = {}
    changed = []
    for k,v in state_dict.items():
        nk = k
        for a,b in rules:
            if nk.startswith(a):
                nk = b + nk[len(a):]
        if nk != k:
            changed.append((k,nk))
        out[nk] = v
    return out, changed

def main():
    ap = argparse.ArgumentParser("Attempt to load a ckpt into current model with mappings")
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--classes', type=int, default=1)
    ap.add_argument('--width', type=float, default=1.0)
    ap.add_argument('--map', action='append', default=[], help='prefix map: old->new (can repeat)')
    args = ap.parse_args()

    from src.models.yolo11_obbpose_td import YOLO11_OBBPOSE_TD
    model = YOLO11_OBBPOSE_TD(num_classes=args.classes, width=args.width)

    ckpt = _safe_torch_load(args.ckpt)
    sd = ckpt.get('model', ckpt)

    sd_mapped, changes = apply_mappings(sd, args.map)
    # Try non-strict load to see which keys load
    missing, unexpected = model.load_state_dict(sd_mapped, strict=False)
    # missing and unexpected are lists in recent PyTorch
    print("=== TRY LOAD REPORT (strict=False) ===")
    print("Mapped rules:", args.map)
    print("Renamed keys:", len(changes))
    print("Missing keys in ckpt for current model:", len(missing))
    print("Unexpected keys in ckpt (not in model):", len(unexpected))
    if missing[:10]:
        print("  examples missing:", missing[:10])
    if unexpected[:10]:
        print("  examples unexpected:", unexpected[:10])

if __name__ == '__main__':
    main()
