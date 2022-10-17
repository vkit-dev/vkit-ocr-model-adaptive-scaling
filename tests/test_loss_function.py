# This project (vkit-x/vkit-open-model) is dual-licensed under commercial and SSPL licenses.
#
# The commercial license gives you the full rights to create and distribute software
# on your own terms without any SSPL license obligations. For more information,
# please see the "LICENSE_COMMERCIAL.txt" file.
#
# This project is also available under Server Side Public License (SSPL).
# The SSPL licensing is ideal for use cases such as open source projects with
# SSPL distribution, student/academic purposes, hobby projects, internal research
# projects without external distribution, or other projects where all SSPL
# obligations can be met. For more information, please see the "LICENSE_SSPL.txt" file.
import torch

from vkit_open_model.loss_function import *


def test_losses():
    for loss_func in [
        WeightedBceWithLogitsLossFunction(),
        L1LossFunction(),
        L1LossFunction(smooth=True),
        DiceLossFunction(),
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
