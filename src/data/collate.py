# src/data/collate.py
from typing import Any, Dict, List, Tuple, Union
import torch

def _can_stack(ts: List[torch.Tensor]) -> bool:
    if not ts or not all(torch.is_tensor(t) for t in ts):
        return False
    s0, d0, dev0 = ts[0].shape, ts[0].dtype, ts[0].device
    return all((t.shape == s0 and t.dtype == d0 and t.device == dev0) for t in ts[1:])

def _empty(shape, dtype, device):
    return torch.zeros(shape, dtype=dtype, device=device)

def collate_obbdet(batch: List[Union[Tuple[Any, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    # --- normalise to dict-per-sample ---
    samples: List[Dict[str, Any]] = []
    first = batch[0]
    if isinstance(first, (tuple, list)) and len(first) == 2:
        for img, tgt in batch:
            d = {'image': img}
            if isinstance(tgt, dict): d.update(tgt)
            else: d['targets'] = tgt
            samples.append(d)
    elif isinstance(first, dict):
        for d in batch:
            d = dict(d)
            if 'img' in d and 'image' not in d:
                d['image'] = d.pop('img')
            samples.append(d)
    else:
        samples = [{'image': x} for x in batch]

    # --- stack images ---
    images = [s['image'] for s in samples]
    images = torch.stack(images, 0) if _can_stack(images) else images  # expect stacked
    device = images.device if torch.is_tensor(images) else torch.device('cpu')
    out: Dict[str, Any] = {'image': images}

    # helper to pull a list and replace missing with empty tensors
    def pull_list(name: str, empty_shape, dtype):
        vals = []
        for s in samples:
            v = s.get(name, None)
            if v is None:
                vals.append(_empty(empty_shape, dtype, device))
            else:
                if torch.is_tensor(v):
                    vals.append(v)
                elif isinstance(v, (list, tuple)):
                    # allow per-sample tensors or already empty
                    if len(v) == 0:
                        vals.append(_empty(empty_shape, dtype, device))
                    else:
                        # keep as is; extractor will tensorise
                        vals.append(v)
                else:
                    vals.append(_empty(empty_shape, dtype, device))
        return vals

    # Provide per-image lists; use empty tensors when absent
    out['bboxes'] = pull_list('bboxes', (0, 5), torch.float32)  # (cx,cy,w,h,ang) expected by loss (ang in radians)
    out['labels'] = pull_list('labels', (0,),  torch.long)
    out['kpts']   = pull_list('kpts',   (0, 2), torch.float32)  # 1 keypoint -> (Ni,2)

    # Keep optional raw YOLO-style 'targets' if your dataset provides it

    tlist = []
    for s in samples:
        t = s.get('targets', None)
        if t is None:
            t = torch.zeros((0, 11), dtype=torch.float32)  # cls + 8 poly + 2 kpt
        elif not torch.is_tensor(t):
            t = torch.as_tensor(t, dtype=torch.float32)
        tlist.append(t)
    out['targets'] = tlist
    out['paths']   = [s.get('paths', None) for s in samples]
    out['meta']    = [s.get('meta',  None) for s in samples]
    return out
