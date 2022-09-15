import torch
from torch.nn import functional as F


class CrossEntropyWithLogitsLossFunction:

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor):
        return F.cross_entropy(pred, gt)
