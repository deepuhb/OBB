# src/engine/dist.py
import os
import torch
import torch.distributed as dist
from datetime import timedelta
from typing import Any, Dict

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def is_main_process() -> bool:
    return get_rank() == 0

def init_distributed(backend: str | None = None, timeout_sec: int = 1800):
    """
    Initialize torch.distributed from torchrun (env://).
    Picks NCCL if CUDA is available; otherwise falls back to GLOO.
    """
    if is_dist():
        return

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        # single-process training; do nothing
        return

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = get_local_rank()

    # Pick backend
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Bind device for CUDA/NCCL
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(seconds=int(timeout_sec)),
        world_size=world_size,
        rank=rank,
    )

def barrier():
    if is_dist():
        dist.barrier()

def synchronize():
    """Alias for barrier() for readability."""
    barrier()

@torch.no_grad()
def reduce_dict(d: Dict[str, Any], average: bool = True) -> Dict[str, Any]:
    """
    All-reduce a dict of scalar tensors across processes. Returns a new dict on all ranks.
    """
    if not is_dist():
        return d
    keys = sorted(d.keys())
    tensors = [torch.as_tensor(d[k], device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)
               for k in keys]
    stack = torch.stack(tensors, dim=0)
    dist.all_reduce(stack, op=dist.ReduceOp.SUM)
    if average:
        stack /= get_world_size()
    out = {k: stack[i].item() for i, k in enumerate(keys)}
    return out
