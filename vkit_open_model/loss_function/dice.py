from typing import Optional

import torch


class DiceLoss:

    def __init__(self, eps: float = 1E-6):
        self.eps = eps

    def __call__(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        if mask is not None:
            pred = pred * mask
            gt *= mask

        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        return loss
