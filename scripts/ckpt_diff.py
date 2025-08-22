#!/usr/bin/env python
from __future__ import annotations
import argparse, sys, os, json
import torch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def _safe_torch_load(path):
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([np.core.multiarray.scalar, np.dtype])
    except Exception:
        pass
    try:
        return torch.load(path, map_location='cpu')  # PyTorch 2.6 default weights_only=True
    except Exception:
        return torch.load(path, map_location='cpu', weights_only=False)

def _build_model(num_classes, width):
    from src.models.yolo11_obbpose_td import YOLO11_OBBPOSE_TD
    return YOLO11_OBBPOSE_TD(num_classes=num_classes, width=width)

def apply_mappings(state_dict, maps):
    # maps: list of "old->new" prefix mappings
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
    ap = argparse.ArgumentParser("Checkpoint vs Model diff")
    ap.add_argument('--ckpt', required=True, help='path to .pt')
    ap.add_argument('--classes', type=int, default=1)
    ap.add_argument('--width', type=float, default=1.0)
    ap.add_argument('--map', action='append', default=[], help='prefix map: old->new (can repeat)')
    args = ap.parse_args()

    ckpt = _safe_torch_load(args.ckpt)
    sd = ckpt.get('model', ckpt)
    model = _build_model(args.classes, args.width)
    msd = model.state_dict()

    # Optionally rename keys
    sd_mapped, changes = apply_mappings(sd, args.map)

    matched = []
    skipped = []
    shape_mismatch = []
    for k,v in sd_mapped.items():
        if k in msd:
            if msd[k].shape == v.shape:
                matched.append(k)
            else:
                shape_mismatch.append((k, tuple(v.shape), tuple(msd[k].shape)))
        else:
            skipped.append(k)

    missing = [k for k in msd.keys() if k not in sd_mapped]

    print("=== CKPT vs MODEL DIFF ===")
    print(f"ckpt tensors: {len(sd)}  mapped: {len(sd_mapped)}")
    print(f"matched: {len(matched)}  shape_mismatch: {len(shape_mismatch)}  skipped(unmapped): {len(skipped)}  missing(in model): {len(missing)}")
    if changes[:10]:
        print("\nExamples of key renames:")
        for a,b in changes[:10]:
            print(f"  {a}  ->  {b}")
    if skipped[:10]:
        print("\nExamples of skipped(unmapped) keys:")
        for k in skipped[:10]:
            print(" ", k)
    if shape_mismatch[:10]:
        print("\nExamples of shape-mismatch keys:")
        for k, s_ckpt, s_model in shape_mismatch[:10]:
            print(f"  {k}: ckpt={s_ckpt} model={s_model}")
    if missing[:10]:
        print("\nExamples of missing model keys:")
        for k in missing[:10]:
            print(" ", k)

    # Suggest a mapping if we detect a systematic extra component (e.g., 'neck.inner.')
    # Heuristic: find longest common prefix among skipped keys, see if removing a token helps.
    # Keep it simple: print top-level prefixes for skipped keys.
    from collections import Counter
    pref_counts = Counter(k.split('.')[0] for k in skipped)
    if pref_counts:
        print("\nSkipped key top-level prefixes:", pref_counts.most_common(10))

if __name__ == '__main__':
    main()
