#!/usr/bin/env python
import argparse, os, sys
from collections import Counter
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', type=int, default=1)
    ap.add_argument('--width', type=float, default=1.0)
    ap.add_argument('--out', type=str, default='/home/dbasavegowda/projects/atb/JaetRob/'
                                               'obbpose11/scripts/model_keys.txt')
    args = ap.parse_args()

    from src.models.yolo11_obbpose_td import YOLO11_OBBPOSE_TD
    m = YOLO11_OBBPOSE_TD(num_classes=args.classes, width=args.width)
    sd = m.state_dict()
    keys = list(sd.keys())
    print(f'Model has {len(keys)} tensors')
    top = Counter(k.split('.')[0] for k in keys).most_common()
    print('Top-level prefixes:', top)
    with open(args.out, 'w') as f:
        for k in keys:
            f.write(k + '\n')
    print(f'Wrote keys to {args.out}')

if __name__ == '__main__':
    main()
