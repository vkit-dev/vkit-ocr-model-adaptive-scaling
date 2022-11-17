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
from torch.nn import functional as F


# https://arxiv.org/abs/2012.15175
class WeightAdaptiveHeatmapRegressionLossFunction:

    def __init__(self, gamma: float = 0.01):
        self.gamma = gamma

    def __call__(
        self,
        # NOTE: should be fed to sigmoid first.
        pred: torch.Tensor,
        gt: torch.Tensor,
    ):
        p = pred
        soft = gt**self.gamma
        weight = soft * (1 - p) + (1 - soft) * p
        l2_loss = F.mse_loss(p, gt, reduction='none')
        return (weight * l2_loss).mean()
