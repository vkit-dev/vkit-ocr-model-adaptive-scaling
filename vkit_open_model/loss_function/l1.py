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
from functools import partial

import torch
from torch.nn import functional as F


class L1LossFunction:

    def __init__(
        self,
        eps: float = 1E-6,
        smooth: bool = False,
        smooth_beta: float = 1.0,
    ):
        self.smooth = smooth
        self.smooth_beta = smooth_beta
        self.eps = eps

    def __call__(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        if not self.smooth:
            torch_loss_func = F.l1_loss
        else:
            torch_loss_func = partial(F.smooth_l1_loss, beta=self.smooth_beta)

        if mask is None:
            return torch_loss_func(pred, gt)
        else:
            loss = torch_loss_func(pred, gt, reduction='none')
            loss *= mask
            return loss.sum() / (mask.sum() + self.eps)
