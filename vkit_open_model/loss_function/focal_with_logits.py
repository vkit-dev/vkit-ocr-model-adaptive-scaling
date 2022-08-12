from typing import Optional

import torch
from torchvision.ops import sigmoid_focal_loss


class FocalWithLogitsLossFunction:

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2,
        eps: float = 1E-6,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def __call__(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        loss = sigmoid_focal_loss(
            inputs=pred,
            targets=gt,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction='mean' if mask is None else 'none',
        )
        if mask is None:
            return loss
        else:
            loss *= mask
            return loss.sum() / (mask.sum() + self.eps)
