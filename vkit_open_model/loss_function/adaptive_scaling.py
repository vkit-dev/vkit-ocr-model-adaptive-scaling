from typing import Tuple

import torch
from vkit.element import Box
import attrs

from .weighted_bce_with_logits import WeightedBceWithLogitsLossFunction
from .focal_with_logits import FocalWithLogitsLossFunction
from .dice import DiceLossFunction
from .l1 import L1LossFunction


@attrs.define
class AdaptiveScalingLossFunctionConifg:
    bce_negative_ratio: float = 3.0
    bce_factor: float = 0.0
    focal_factor: float = 5.0
    dice_factor: float = 1.0
    l1_factor: float = 1.0
    downsampled_score_map_min: float = 1.1
    scale_feature_min: float = 1.1


class AdaptiveScalingLossFunction:

    def __init__(self, config: AdaptiveScalingLossFunctionConifg):
        self.config = config

        # Mask.
        self.weighted_bce_with_logits = WeightedBceWithLogitsLossFunction(
            negative_ratio=config.bce_negative_ratio,
        )
        self.focal_with_logits = FocalWithLogitsLossFunction()
        self.dice = DiceLossFunction()

        # Scale.
        self.l1 = L1LossFunction(smooth=True)

    def __call__(
        self,
        mask_feature: torch.Tensor,
        scale_feature: torch.Tensor,
        downsampled_mask: torch.Tensor,
        downsampled_score_map: torch.Tensor,
        downsampled_shape: Tuple[int, int],
        downsampled_core_box: Box,
    ) -> torch.Tensor:
        # (B, 1, H, W)
        assert mask_feature.shape == scale_feature.shape
        assert mask_feature.shape[1:] == (1, *downsampled_shape)

        # (B, H, W)
        mask_feature = torch.squeeze(mask_feature, dim=1)
        scale_feature = torch.squeeze(scale_feature, dim=1)

        # (B, DH, DW)
        dc_box = downsampled_core_box
        mask_feature = mask_feature[:, dc_box.up:dc_box.down + 1, dc_box.left:dc_box.right + 1]
        scale_feature = scale_feature[:, dc_box.up:dc_box.down + 1, dc_box.left:dc_box.right + 1]

        loss = 0.0

        # Mask.
        if self.config.bce_factor > 0.0:
            loss += self.config.bce_factor * self.weighted_bce_with_logits(
                pred=mask_feature,
                gt=downsampled_mask,
            )

        if self.config.focal_factor > 0.0:
            loss += self.config.focal_factor * self.focal_with_logits(
                pred=mask_feature,
                gt=downsampled_mask,
            )

        if self.config.dice_factor > 0.0:
            loss += self.config.dice_factor * self.dice(
                pred=torch.sigmoid(mask_feature),
                gt=downsampled_mask,
            )

        # Scale.
        if self.config.l1_factor > 0.0:
            # NOTE: critical mask!
            l1_mask = ((scale_feature > self.config.scale_feature_min)
                       & (downsampled_score_map > self.config.downsampled_score_map_min)
                       & downsampled_mask.bool()).float()
            scale_feature = torch.clamp(
                scale_feature,
                min=self.config.scale_feature_min,
            )
            downsampled_score_map = torch.clamp(
                downsampled_score_map,
                min=self.config.downsampled_score_map_min,
            )
            # Log space + smooth L1 to model the relative scale difference.
            loss += self.config.l1_factor * self.l1(
                pred=torch.log(scale_feature),
                gt=torch.log(downsampled_score_map),
                mask=l1_mask,
            )

        assert not isinstance(loss, float)
        return loss
