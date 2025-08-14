# src/utils/distrib.py
from __future__ import annotations
import os
import torch
import torch.distributed as dist

IS_DISTRIBUTED = False
RANK = 0
WORLD_SIZE = 1
LOCAL_RANK = 0

def init(backend: str = "nccl") -> tuple[int, int, bool]:
    """
    Initialize torch.distributed *only if* launched with torchrun (env vars present).
    Returns (rank, world_size, is_distributed).
    """
    global IS_DISTRIBUTED, RANK, WORLD_SIZE, LOCAL_RANK

    world_size_env = os.environ.get("WORLD_SIZE")
    rank_env = os.environ.get("RANK")
    local_rank_env = os.environ.get("LOCAL_RANK")

    WORLD_SIZE = int(world_size_env) if world_size_env else 1
    RANK = int(rank_env) if rank_env else 0
    LOCAL_RANK = int(local_rank_env) if local_rank_env else 0

    if WORLD_SIZE > 1 and rank_env is not None:
        dist.init_process_group(backend=backend, init_method="env://")
        IS_DISTRIBUTED = True
        if torch.cuda.is_available() and backend == "nccl":
            torch.cuda.set_device(LOCAL_RANK)
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
    else:
        IS_DISTRIBUTED = False
        RANK, WORLD_SIZE, LOCAL_RANK = 0, 1, 0

    return RANK, WORLD_SIZE, IS_DISTRIBUTED

def is_main_process() -> bool:
    return (not IS_DISTRIBUTED) or (RANK == 0)

def maybe_barrier():
    if IS_DISTRIBUTED and dist.is_initialized():
        dist.barrier()

def cleanup():
    if IS_DISTRIBUTED and dist.is_initialized():
        dist.destroy_process_group()
