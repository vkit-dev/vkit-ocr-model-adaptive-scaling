from typing import Optional
from functools import partial

import torch
from torch.nn import functional as F


class L1LossFunction:

    def __init__(
        self,
        eps: float = 1E-6,
        smooth: bool = False,
        smooth_beta: float = 1.0,
    ):
        self.smooth = smooth
        self.smooth_beta = smooth_beta
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
            torch_loss_func = partial(F.smooth_l1_loss, beta=self.smooth_beta)

        if mask is None:
            return torch_loss_func(pred, gt)
        else:
            loss = torch_loss_func(pred, gt, reduction='none')
            loss *= mask
            return loss.sum() / (mask.sum() + self.eps)
