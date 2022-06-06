from typing import List
from enum import Enum, unique

import torch
from torch import nn

from .convnext import ConvNext
from .upernext import UperNext


@unique
class AdaptiveScalingSize(Enum):
    TINY = 'tiny'
    SMALL = 'small'
    BASE = 'base'
    LARGE = 'large'


class AdaptiveScaling(nn.Module):

    def __init__(self, size: AdaptiveScalingSize, stem_use_pconv2x2: bool):
        super().__init__()

        if size == AdaptiveScalingSize.TINY:
            backbone_creator = ConvNext.create_tiny
            head_creator = UperNext.create_tiny
            head_mid_channels = 32

        elif size == AdaptiveScalingSize.SMALL:
            backbone_creator = ConvNext.create_small
            head_creator = UperNext.create_small
            head_mid_channels = 32

        elif size == AdaptiveScalingSize.BASE:
            backbone_creator = ConvNext.create_base
            head_creator = UperNext.create_base
            head_mid_channels = 64

        elif size == AdaptiveScalingSize.LARGE:
            backbone_creator = ConvNext.create_large
            head_creator = UperNext.create_large
            head_mid_channels = 128

        else:
            raise NotImplementedError()

        self.backbone = backbone_creator(stem_use_pconv2x2=stem_use_pconv2x2)
        self.mask_head = head_creator(out_channels=1, mid_channels=head_mid_channels)
        self.scale_head = head_creator(out_channels=1, mid_channels=head_mid_channels)

    @staticmethod
    def create_tiny(stem_use_pconv2x2: bool = False):
        return AdaptiveScaling(
            size=AdaptiveScalingSize.TINY,
            stem_use_pconv2x2=stem_use_pconv2x2,
        )

    @staticmethod
    def create_small(stem_use_pconv2x2: bool = False):
        return AdaptiveScaling(
            size=AdaptiveScalingSize.SMALL,
            stem_use_pconv2x2=stem_use_pconv2x2,
        )

    @staticmethod
    def create_base(stem_use_pconv2x2: bool = False):
        return AdaptiveScaling(
            size=AdaptiveScalingSize.BASE,
            stem_use_pconv2x2=stem_use_pconv2x2,
        )

    @staticmethod
    def create_large(stem_use_pconv2x2: bool = False):
        return AdaptiveScaling(
            size=AdaptiveScalingSize.LARGE,
            stem_use_pconv2x2=stem_use_pconv2x2,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # type: ignore
        features = self.backbone(x)
        mask_feature = self.mask_head(features)
        scale_feature = self.scale_head(features)
        return [mask_feature, scale_feature]
