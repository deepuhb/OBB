# src/training/lr_utils.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def _get_param_groups(optim: Optimizer):
    # Return list of param_groups
    return optim.param_groups

def _set_lr(optim: Optimizer, lrs: List[float]):
    groups = _get_param_groups(optim)
    assert len(groups) == len(lrs)
    for g, lr in zip(groups, lrs):
        g["lr"] = float(lr)

def _set_momentum(optim: Optimizer, moms: List[float]):
    for g, m in zip(optim.param_groups, moms):
        if "momentum" in g:
            g["momentum"] = float(m)
        elif "betas" in g and isinstance(g["betas"], tuple):
            beta1, beta2 = g["betas"]
            g["betas"] = (float(m), beta2)

@dataclass
class WarmupState:
    lr0: List[float]
    warmup_lrs: List[float]
    mom0: List[float]
    warmup_moms: List[float]
    nw: int  # warmup iterations

def build_yolo_scheduler(
    optimizer: Optimizer,
    epochs: int,
    lr0: float = 0.002,
    lrf: float = 0.01,
    warmup_epochs: float = 3.0,
    momentum: float = 0.937,
    warmup_momentum: float = 0.8,
    iters_per_epoch: Optional[int] = None,
):
    '''
    Returns (scheduler, warmup_state, warmup_step_fn).

    - Cosine decay factor: f(e) = ((1 + cos(pi * e/epochs)) / 2) * (1 - lrf) + lrf
      so lr(e) = lr0 * f(e), with lr(0) = lr0, lr(epochs) = lr0 * lrf.
    - Warmup: linear per-iteration warmup over `nw = warmup_epochs * iters_per_epoch` steps.
      LR: 0 -> lr0, Momentum: warmup_momentum -> momentum.
    '''
    assert epochs > 0, "epochs must be > 0"

    # setup base lrs for each param group
    for g in optimizer.param_groups:
        g.setdefault("initial_lr", lr0)  # unify
        # if trainer already set different lrs per group, we respect them as base
        if "lr" not in g or g["lr"] == 0.0:
            g["lr"] = lr0

    # cosine scheduler per-epoch
    def lf(epoch: int):
        x = epoch / max(1, epochs)
        return ((1 + math.cos(math.pi * x)) / 2) * (1 - lrf) + lrf

    scheduler = LambdaLR(optimizer, lr_lambda=lf)

    # warmup setup
    if warmup_epochs and warmup_epochs > 0 and iters_per_epoch and iters_per_epoch > 0:
        nw = int(round(warmup_epochs * iters_per_epoch))
    else:
        nw = 0

    base_lrs = [g["lr"] for g in optimizer.param_groups]
    base_moms = []
    for g in optimizer.param_groups:
        if "momentum" in g:
            base_moms.append(g["momentum"])
        elif "betas" in g and isinstance(g["betas"], tuple):
            base_moms.append(g["betas"][0])
        else:
            base_moms.append(momentum)

    warmup_state = WarmupState(
        lr0=base_lrs,
        warmup_lrs=[0.0 for _ in base_lrs],  # start at 0 for all
        mom0=base_moms,
        warmup_moms=[warmup_momentum for _ in base_moms],
        nw=nw,
    )

    def warmup_step(ni: int):
        '''
        Per-iteration warmup. Call with global iteration index (starting at 0).
        Linearly increases LR from 0 -> base_lr and momentum from warmup_momentum -> momentum.
        '''
        if warmup_state.nw <= 0 or ni >= warmup_state.nw:
            return
        # linear ramp
        frac = (ni + 1) / warmup_state.nw
        lrs = [wl + frac * (bl - wl) for wl, bl in zip(warmup_state.warmup_lrs, warmup_state.lr0)]
        moms = [wm + frac * (bm - wm) for wm, bm in zip(warmup_state.warmup_moms, warmup_state.mom0)]
        _set_lr(optimizer, lrs)
        _set_momentum(optimizer, moms)

    return scheduler, warmup_state, warmup_step

# -------- Optional: safe exp helper for decode-time IoU target --------
def safe_exp(x: torch.Tensor, min_val: float = -6.0, max_val: float = 6.0) -> torch.Tensor:
    '''
    Clamp then exp to avoid fp16 overflow during very early training.
    exp(6) is about 403; multiplied by stride keeps sizes in a sane range.
    '''
    return torch.exp(x.clamp(min=min_val, max=max_val))
