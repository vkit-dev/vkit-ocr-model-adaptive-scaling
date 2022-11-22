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


class L2LossFunction:

    def __init__(self, eps: float = 1E-6):
        self.eps = eps

    def __call__(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        if mask is None:
            return F.mse_loss(pred, gt)
        else:
            loss = F.mse_loss(pred, gt, reduction='none')
            loss *= mask
            return loss.sum() / (mask.sum() + self.eps)
