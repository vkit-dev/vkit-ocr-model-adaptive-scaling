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
from typing import Tuple, Dict, Mapping
from enum import Enum, unique
import logging

import torch
from torch import nn
import attrs

from .convnext import ConvNext
from .fpn import FpnNeck, FpnHead
from .upernext import UperNextNeck, UperNextHead

logger = logging.getLogger(__name__)


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
    init_char_height_output_bias: float = 8.0


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
        self.rough_char_height_head = nn.Sequential(
            head_creator(
                in_channels=neck_out_channels,
                out_channels=1,
                upsampling_factor=2,
                init_output_bias=config.init_char_height_output_bias,
            ),
            # Force predicting positive value.
            nn.Softplus(),
        )

        # Neck for precise detection.
        self.precise_neck = neck_creator(
            in_channels_group=self.backbone.in_channels_group,
            out_channels=neck_out_channels,
        )

        # 5 heads, 2x upsampling, leading to 2x E2E downsampling.
        self.precise_char_mask_head = head_creator(
            in_channels=neck_out_channels,
            out_channels=1,
            upsampling_factor=2,
        )
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
        rough_char_height_feature = self.rough_char_height_head(rough_neck_feature)

        return rough_char_mask_feature, rough_char_height_feature

    @torch.jit.export  # type: ignore
    def forward_precise(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:  # type: ignore
        feature = self.backbone(x)

        precise_neck_feature = self.precise_neck(feature)
        precise_char_mask_feature = self.precise_char_mask_head(precise_neck_feature)
        precise_char_prob_feature = self.precise_char_prob_head(precise_neck_feature)
        precise_char_up_left_corner_offset_feature = \
            self.precise_char_up_left_corner_offset_head(precise_neck_feature)
        precise_char_corner_angle_feature = \
            self.precise_char_corner_angle_head(precise_neck_feature)
        precise_char_corner_distance_feature = \
            self.precise_char_corner_distance_head(precise_neck_feature)

        return (
            precise_char_mask_feature,
            precise_char_prob_feature,
            precise_char_up_left_corner_offset_feature,
            precise_char_corner_angle_feature,
            precise_char_corner_distance_feature,
        )

    @classmethod
    def debug_get_rough_name_to_grad(cls, model: torch.nn.Module):
        rough_name_to_grad: Dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            assert name not in rough_name_to_grad
            rough_name_to_grad[name] = parameter.grad.cpu().clone()  # type: ignore
        return rough_name_to_grad

    @classmethod
    def debug_get_precise_name_to_grad(
        cls,
        model: torch.nn.Module,
        rough_name_to_grad: Mapping[str, torch.Tensor],
    ):
        precise_name_to_grad: Dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if name not in rough_name_to_grad:
                continue
            assert name not in precise_name_to_grad
            precise_name_to_grad[name] = \
                parameter.grad.cpu() - rough_name_to_grad[name]  # type: ignore
        return precise_name_to_grad

    @classmethod
    def debug_inspect_name_to_grad(
        cls,
        rough_name_to_grad: Mapping[str, torch.Tensor],
        precise_name_to_grad: Mapping[str, torch.Tensor],
    ):
        names = sorted(set(rough_name_to_grad) & set(precise_name_to_grad))

        rough_abs_grads = torch.abs(
            torch.cat([rough_name_to_grad[name].view(-1) for name in names])
        )
        rough_abs_grads_mean = float(torch.mean(rough_abs_grads))
        rough_abs_grads_std = float(torch.std(rough_abs_grads))
        logger.info(
            f'rough_abs_grads_mean = {rough_abs_grads_mean}, '
            f'rough_abs_grads_std = {rough_abs_grads_std}'
        )

        precise_abs_grads = torch.abs(
            torch.cat([precise_name_to_grad[name].view(-1) for name in names])
        )
        precise_abs_grads_mean = float(torch.mean(precise_abs_grads))
        precise_abs_grads_std = float(torch.std(precise_abs_grads))
        logger.info(
            f'precise_abs_grads_mean = {precise_abs_grads_mean}, '
            f'precise_abs_grads_std = {precise_abs_grads_std}'
        )

        logger.info(
            'rough_abs_grads_mean / precise_abs_grads_mean = '
            f'{rough_abs_grads_mean / (precise_abs_grads_mean + 1E-15)}'
        )
