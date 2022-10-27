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
