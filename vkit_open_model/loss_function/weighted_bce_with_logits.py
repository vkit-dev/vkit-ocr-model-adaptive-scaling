from typing import Optional

import torch
from torch.nn import functional as F


class WeightedBceWithLogitsLossFunction:

    def __init__(self, negative_ratio: float = 3.0, eps: float = 1E-6):
        self.negative_ratio = negative_ratio
        self.eps = eps

    def __call__(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        positive_mask = gt
        negative_mask = (1 - gt)
        if mask is not None:
            positive_mask *= mask
            negative_mask *= mask

        positive_mask = positive_mask.byte()
        positive_count = int(positive_mask.long().sum())
        positive_mask = positive_mask.float()

        negative_mask = negative_mask.byte()
        negative_count = int(negative_mask.long().sum())
        negative_count = min(round(positive_count * self.negative_ratio), negative_count)
        negative_mask = negative_mask.float()

        loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')

        positive_loss = loss * positive_mask

        negative_loss = loss * negative_mask
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        weighted_loss = positive_loss.sum() + negative_loss.sum()
        weighted_loss /= (positive_count + negative_count + self.eps)
        return weighted_loss
