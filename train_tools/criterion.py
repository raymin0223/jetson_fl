import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ["OverhaulLoss"]


class OverhaulLoss(nn.Module):
    """Loss function (Local Objective)"""

    def __init__(self, mode="CE", num_classes=10, temp=1, beta=1, lam=0):
        super(OverhaulLoss, self).__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.temp = temp
        self.beta = beta
        self.lam = lam

    def forward(self, logits, target, t_logits=None):
        """Calculate loss by given modes"""

        loss = torch.zeros(logits.size(0)).to(str(target.device))  # initialize loss

        # Cross-Entropy Loss (FedAvg Baseline)
        if self.mode == "CE":
            loss += F.cross_entropy(logits / self.temp, target, reduction="none")

        loss = loss.mean()

        return loss