# multi_noise_loss.py
import torch
import torch.nn as nn

class MultiNoiseLoss(nn.Module):
    """
    Simple homoscedastic uncertainty weighting over multiple losses.
    Kept identical to the working version, but with a tiny numerical guard.
    """
    def __init__(self, num_losses: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        total = 0.0
        for i, loss_i in enumerate(losses):
            precision = torch.exp(-self.log_vars[i].clamp(-10, 10))
            total = total + precision * loss_i + self.log_vars[i]
        return 0.5 * total
