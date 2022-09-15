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

        # Shared backbone, 4x downsampling.
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

        # Neck for rough detection.
        self.rough_neck = neck_creator(
            in_channels_group=self.backbone.in_channels_group,
            out_channels=neck_out_channels,
        )

        # 2 heads, 2x upsampling, leading to 2x E2E downsampling.
        self.rough_char_mask_head = head_creator(
            in_channels=neck_out_channels,
            out_channels=1,
            upsampling_factor=2,
        )
        self.rough_char_scale_head = nn.Sequential(
            head_creator(
                in_channels=neck_out_channels,
                out_channels=1,
                upsampling_factor=2,
                init_output_bias=config.init_scale_output_bias,
            ),
            # Force predicting positive value.
            nn.Softplus(),
        )

        # Neck for precise detection.
        self.precise_neck = neck_creator(
            in_channels_group=self.backbone.in_channels_group,
            out_channels=neck_out_channels,
        )

        # 4 heads, 2x upsampling, leading to 2x E2E downsampling.
        self.precise_char_prob_head = head_creator(
            in_channels=neck_out_channels,
            out_channels=1,
            upsampling_factor=2,
        )
        self.precise_char_up_left_corner_offset_head = head_creator(
            in_channels=neck_out_channels,
            out_channels=2,
            upsampling_factor=2,
        )
        self.precise_char_corner_angle_head = head_creator(
            in_channels=neck_out_channels,
            out_channels=4,
            upsampling_factor=2,
        )
        self.precise_char_corner_distance_head = nn.Sequential(
            head_creator(
                in_channels=neck_out_channels,
                out_channels=3,
                upsampling_factor=2,
            ),
            # Force predicting positive value.
            nn.Softplus(),
        )

    @torch.jit.export  # type: ignore
    def forward_rough(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        feature = self.backbone(x)

        rough_neck_feature = self.rough_neck(feature)
        rough_char_mask_feature = self.rough_char_mask_head(rough_neck_feature)
        rough_char_scale_feature = self.rough_char_scale_head(rough_neck_feature)

        return rough_char_mask_feature, rough_char_scale_feature

    @torch.jit.export  # type: ignore
    def forward_precise(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore
        feature = self.backbone(x)

        precise_neck_feature = self.precise_neck(feature)
        precise_char_prob_feature = self.precise_char_prob_head(precise_neck_feature)
        precise_char_up_left_corner_offset_feature = \
            self.precise_char_up_left_corner_offset_head(precise_neck_feature)
        precise_char_corner_angle_feature = \
            self.precise_char_corner_angle_head(precise_neck_feature)
        precise_char_corner_distance_feature = \
            self.precise_char_corner_distance_head(precise_neck_feature)

        return (
            precise_char_prob_feature,
            precise_char_up_left_corner_offset_feature,
            precise_char_corner_angle_feature,
            precise_char_corner_distance_feature,
        )
