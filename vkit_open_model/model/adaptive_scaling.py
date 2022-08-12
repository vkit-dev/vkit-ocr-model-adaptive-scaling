from typing import Tuple
from enum import Enum, unique

import torch
from torch import nn
import attrs

from .convnext import ConvNext
from .fpn import FpnNeck, FpnHead
from .upernext import UperNextNeck, UperNextHead


@unique
class AdaptiveScalingSize(Enum):
    TINY = 'tiny'
    SMALL = 'small'
    BASE = 'base'
    LARGE = 'large'


@unique
class AdaptiveScalingNeckHeadType(Enum):
    FPN = 'fpn'
    UPERNEXT = 'upernext'


@attrs.define
class AdaptiveScalingConfig:
    size: AdaptiveScalingSize = AdaptiveScalingSize.SMALL
    neck_head_type: AdaptiveScalingNeckHeadType = AdaptiveScalingNeckHeadType.FPN
    init_scale_output_bias: float = 8.0


class AdaptiveScaling(nn.Module):

    def __init__(self, config: AdaptiveScalingConfig):
        super().__init__()

        if config.size == AdaptiveScalingSize.TINY:
            backbone_creator = ConvNext.create_tiny
        elif config.size == AdaptiveScalingSize.SMALL:
            backbone_creator = ConvNext.create_small
        elif config.size == AdaptiveScalingSize.BASE:
            backbone_creator = ConvNext.create_base
        elif config.size == AdaptiveScalingSize.LARGE:
            backbone_creator = ConvNext.create_large
        else:
            raise NotImplementedError()

        # 4x downsampling.
        self.backbone = backbone_creator()

        if config.neck_head_type == AdaptiveScalingNeckHeadType.FPN:
            neck_creator = FpnNeck
            head_creator = FpnHead
        elif config.neck_head_type == AdaptiveScalingNeckHeadType.UPERNEXT:
            neck_creator = UperNextNeck
            head_creator = UperNextHead
        else:
            raise NotImplementedError()

        neck_out_channels = self.backbone.in_channels_group[-2]

        # Shared neck.
        self.neck = neck_creator(
            in_channels_group=self.backbone.in_channels_group,
            out_channels=neck_out_channels,
        )

        # Two heads, 2x upsampling, leading to 2x E2E downsampling.
        self.mask_head = head_creator(
            in_channels=neck_out_channels,
            out_channels=1,
            upsampling_factor=2,
        )
        self.scale_head = head_creator(
            in_channels=neck_out_channels,
            out_channels=1,
            upsampling_factor=2,
            init_output_bias=config.init_scale_output_bias,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        features = self.backbone(x)
        neck_feature = self.neck(features)
        mask_feature = self.mask_head(neck_feature)
        scale_feature = self.scale_head(neck_feature)
        return mask_feature, scale_feature
