# src/utils/model_schema.py
import torch, hashlib, json, inspect
from collections import OrderedDict

def schema_signature(model: torch.nn.Module):
    keys = list(model.state_dict().keys())
    # stable signature independent of values
    sig = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()[:16]
    # capture file origins of key modules that often drift
    origins = {}
    for name in ["backbone", "neck", "head", "kpt_head"]:
        mod = getattr(model, name, None)
        if mod is not None:
            try:
                origins[name] = inspect.getsourcefile(mod.__class__) or "<?>"
            except Exception:
                origins[name] = "<?>"
    return sig, keys, origins

def print_schema(model, tag="MODEL"):
    sig, keys, origins = schema_signature(model)
    print(f"[SCHEMA {tag}] sig={sig} params={len(keys)}")
    for k in ("backbone","neck","head","kpt_head"):
        if k in origins:
            print(f"[SCHEMA {tag}] {k} from: {origins[k]}")
    # show 10 example keys for neck/head
    prefix_keys = [k for k in keys if k.startswith("neck.") or k.startswith("head.")]
    for k in prefix_keys[:10]:
        print(f"[SCHEMA {tag}] ex: {k}")
