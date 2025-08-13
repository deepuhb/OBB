# src/engine/dist.py
import os
import torch
import torch.distributed as dist
from datetime import timedelta

def is_dist():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist() else 0

def is_main_process():
    return get_rank() == 0

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def init_distributed(backend="nccl", timeout_sec=1800):
    """
    Initialize torch.distributed using environment variables set by torchrun.
    """
    if is_dist():
        return

    # Only init if launched with torchrun (env://)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Set device before init when using CUDA/NCCL
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=int(timeout_sec)),
            world_size=world_size,
            rank=rank,
        )
    # else: running single-process; do nothing

def barrier():
    if is_dist():
        dist.barrier()
