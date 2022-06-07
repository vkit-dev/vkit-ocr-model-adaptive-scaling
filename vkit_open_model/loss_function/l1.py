from typing import Optional

import torch
from torch.nn import functional as F


class L1LossFunction:

    def __init__(self, eps: float = 1E-6):
        self.eps = eps

    def __call__(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        delta = torch.abs(pred - gt)
        if mask is not None:
            loss = (delta * mask).sum() / (mask.sum() + self.eps)
        else:
            loss = delta.mean()
        return loss
