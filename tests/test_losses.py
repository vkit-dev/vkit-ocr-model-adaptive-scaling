import torch

from vkit_open_model.loss_function import WeightedBceWithLogitsLossFunction


def test_weighted_bce_with_logits_loss():
    loss_func = WeightedBceWithLogitsLossFunction()

    pred = torch.randn(100, requires_grad=True)
    gt = torch.empty(100).random_(0, 2)
    loss = loss_func(pred, gt)
    loss.backward()
