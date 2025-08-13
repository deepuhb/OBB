# src/config/config.py
import os, yaml
from types import SimpleNamespace

def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_to_ns(x) for x in d]
    else:
        return d

def load_config(path: str):
    path = os.path.expanduser(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = _to_ns(raw)
    cfg._raw = raw  # keep raw dict if needed
    return cfg
