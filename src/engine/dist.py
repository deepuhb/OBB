# src/engine/dist.py
import os
from datetime import timedelta
from typing import Any, Dict

import torch
import torch.distributed as dist


# -------------------------
# Introspection helpers
# -------------------------
def ddp_is_enabled() -> bool:
    """True if torch.distributed has been initialized (DDP)."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if ddp_is_enabled() else 0


def get_world_size() -> int:
    return dist.get_world_size() if ddp_is_enabled() else 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


# -------------------------
# Setup / teardown
# -------------------------
def setup_ddp(backend: str | None = None, timeout_sec: int = 1800):
    """
    Initialize torch.distributed from torchrun (env://). Idempotent.
    Chooses NCCL if CUDA is available, otherwise Gloo. Also sets the CUDA device
    using LOCAL_RANK for NCCL.
    """
    if ddp_is_enabled():
        return

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        # single-process training; nothing to do
        return

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = get_local_rank()

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(seconds=int(timeout_sec)),
        world_size=world_size,
        rank=rank,
    )


def cleanup_ddp():
    """Destroy the default process group if initialized (idempotent)."""
    if ddp_is_enabled():
        dist.destroy_process_group()


# -------------------------
# Utilities
# -------------------------
def barrier():
    if ddp_is_enabled():
        dist.barrier()


def synchronize():
    """Alias for barrier() for readability."""
    barrier()


@torch.no_grad()
def reduce_dict(d: Dict[str, Any], average: bool = True) -> Dict[str, Any]:
    """
    All-reduce a dict of scalar tensors across processes.
    Returns a NEW dict on all ranks.
    """
    if not ddp_is_enabled():
        return d

    keys = sorted(d.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = [torch.as_tensor(d[k], device=device, dtype=torch.float32) for k in keys]
    stack = torch.stack(tensors, dim=0)
    dist.all_reduce(stack, op=dist.ReduceOp.SUM)
    if average:
        stack /= get_world_size()
    out = {k: stack[i].item() for i, k in enumerate(keys)}
    return out


# -------------------------
# Back-compat shim (older names used earlier in the project)
# -------------------------
# Older code may import these names; keep them as aliases.
is_dist = ddp_is_enabled
init_distributed = setup_ddp
synchronize_processes = synchronize