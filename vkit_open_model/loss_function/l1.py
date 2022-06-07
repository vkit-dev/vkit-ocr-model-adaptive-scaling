from typing import Optional

import torch
from torch.nn import functional as F


class L1LossFunction:

    def __init__(self, eps: float = 1E-6, smooth: bool = False):
        self.smooth = smooth
        self.eps = eps

    def __call__(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        if not self.smooth:
            torch_loss_func = F.l1_loss
        else:
            torch_loss_func = F.smooth_l1_loss

        if mask is None:
            return torch_loss_func(pred, gt)
        else:
            loss = torch_loss_func(pred, gt, reduction='none')
            return loss.sum() / (mask.sum() + self.eps)
