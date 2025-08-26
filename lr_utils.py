# src/training/lr_utils.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable, Dict, Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _get_param_groups(optim: Optimizer):
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
    lr0: List[float]            # target/base LR per group (end of warmup)
    warmup_lrs: List[float]     # starting LR per group (usually 0.0)
    mom0: List[float]           # target/base momentum (or beta1) per group
    warmup_moms: List[float]    # starting momentum per group
    nw: int                     # warmup iterations


def build_yolo_scheduler(
    optimizer: Optimizer,
    epochs: int,
    lr0: float = 0.01,
    lrf: float = 0.01,
    warmup_epochs: float = 3.0,
    momentum: float = 0.937,
    warmup_momentum: float = 0.8,
    iters_per_epoch: Optional[int] = None,
) -> Tuple[LambdaLR, WarmupState, Callable[[int], None]]:
    \"\"\"
    Returns (scheduler, warmup_state, warmup_step_fn). Compatible with existing callers.

    - Cosine decay per epoch: f(e) = ((1 + cos(pi * e/epochs)) / 2) * (1 - lrf) + lrf
      so lr(e) = base_lr * f(e), with lr(0) = base_lr, lr(epochs) = base_lr * lrf.
    - Warmup (per-iteration): linear over `nw = warmup_epochs * iters_per_epoch` steps.
      LR: start -> base (default start=0.0) ; Momentum: warmup_momentum -> momentum.
    - Respects per-group base LR if you set different LRs on param_groups before calling this.
    \"\"\"
    assert epochs > 0, "epochs must be > 0"

    # Fallback if an IterableDataset hides length
    if iters_per_epoch is None or iters_per_epoch <= 0:
        iters_per_epoch = 1000
    nw = int(round(max(0.0, float(warmup_epochs)) * float(iters_per_epoch)))

    # --- Capture per-group base LR and momentum BEFORE creating the scheduler ---
    base_lrs: List[float] = []
    base_moms: List[float] = []
    for g in optimizer.param_groups:
        # Respect a pre-set 'initial_lr' or current 'lr'; default to lr0
        base_lr = float(g.get("initial_lr", g.get("lr", lr0)))
        g["initial_lr"] = base_lr  # what LambdaLR will use as base
        base_lrs.append(base_lr)

        if "momentum" in g:
            base_moms.append(float(g["momentum"]))
        elif "betas" in g and isinstance(g["betas"], tuple):
            base_moms.append(float(g["betas"][0]))
        else:
            base_moms.append(float(momentum))

    # Make sure optimizer LRs reflect the base for scheduler init
    _set_lr(optimizer, base_lrs)

    # --- Cosine schedule across epochs (on top of base_lrs) ---
    def lf(epoch: int):
        x = epoch / max(1, epochs)
        return ((1 + math.cos(math.pi * x)) / 2) * (1 - lrf) + lrf

    scheduler = LambdaLR(optimizer, lr_lambda=lf)

    # --- Now set warmup *starting* state (usually 0 LR, warmup_momentum) ---
    warmup_lrs = [0.0 for _ in base_lrs]  # 0 -> base
    warmup_moms = [float(warmup_momentum) for _ in base_moms]
    _set_lr(optimizer, warmup_lrs)
    _set_momentum(optimizer, warmup_moms)

    warmup_state = WarmupState(
        lr0=base_lrs,
        warmup_lrs=warmup_lrs,
        mom0=base_moms,
        warmup_moms=warmup_moms,
        nw=nw,
    )

    def warmup_step(ni: int) -> None:
        \"\"\"
        Per-iteration warmup. Call with global iteration index starting at 0.
        Linearly increases LR from warmup_lrs -> base_lrs and momentum from warmup_moms -> mom0.
        \"\"\"
        if warmup_state.nw <= 0 or ni >= warmup_state.nw:
            return
        frac = float(ni + 1) / float(warmup_state.nw)  # in (0,1]
        lrs = [wl + frac * (bl - wl) for wl, bl in zip(warmup_state.warmup_lrs, warmup_state.lr0)]
        moms = [wm + frac * (bm - wm) for wm, bm in zip(warmup_state.warmup_moms, warmup_state.mom0)]
        _set_lr(optimizer, lrs)
        _set_momentum(optimizer, moms)

    return scheduler, warmup_state, warmup_step


# -------- Optional: safe exp helper for decode-time IoU target --------
def safe_exp(x: torch.Tensor, min_val: float = -6.0, max_val: float = 6.0) -> torch.Tensor:
    \"\"\"
    Clamp then exp to avoid fp16 overflow during very early training.
    exp(6) is about 403; multiplied by stride keeps sizes in a sane range.
    \"\"\"
    return torch.exp(x.clamp(min=min_val, max=max_val))