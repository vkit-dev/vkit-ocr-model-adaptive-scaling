import torch

from vkit_open_model.loss_function import *


def test_weighted_bce_with_logits_loss():
    loss_func = WeightedBceWithLogitsLossFunction()

    pred = torch.randn(100, requires_grad=True)
    gt = torch.empty(100).random_(0, 2)
    loss = loss_func(pred, gt)
    loss.backward()

    pred = torch.randn(100, requires_grad=True)
    gt = torch.empty(100).random_(0, 2)
    mask = torch.empty(100).random_(0, 2)
    loss = loss_func(pred, gt, mask)
    loss.backward()


def test_l1_loss():
    loss_func = L1LossFunction()

    pred = torch.randn(100, requires_grad=True)
    gt = torch.empty(100).random_(0, 2)
    loss = loss_func(pred, gt)
    loss.backward()

    pred = torch.randn(100, requires_grad=True)
    gt = torch.empty(100).random_(0, 2)
    mask = torch.empty(100).random_(0, 2)
    loss = loss_func(pred, gt, mask)
    loss.backward()
