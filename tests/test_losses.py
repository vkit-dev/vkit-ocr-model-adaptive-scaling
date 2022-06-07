import torch

from vkit_open_model.loss_function import *


def test_losses():
    for loss_func in [
        WeightedBceWithLogitsLossFunction(),
        L1LossFunction(),
        L1LossFunction(smooth=True),
        DiceLoss(),
    ]:
        pred = torch.randn(100, requires_grad=True)
        gt = torch.empty(100).random_(0, 2)
        loss = loss_func(pred, gt)
        loss.backward()

        pred = torch.randn(100, requires_grad=True)
        gt = torch.empty(100).random_(0, 2)
        mask = torch.empty(100).random_(0, 2)
        loss = loss_func(pred, gt, mask)
        loss.backward()
