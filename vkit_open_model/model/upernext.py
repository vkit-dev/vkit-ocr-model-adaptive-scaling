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
from typing import Sequence, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F

from . import helper


def build_conv1x1_block(in_channels: int, out_channels: int, no_ln: bool = False):
    modules: List[torch.nn.Module] = [
        helper.permute_bchw_to_bhwc(),
        helper.conv1x1(in_channels=in_channels, out_channels=out_channels),
    ]

    if not no_ln:
        modules.append(helper.ln(in_channels=out_channels))

    modules.extend([
        helper.permute_bhwc_to_bchw(),
        helper.gelu(),
    ])

    return nn.Sequential(*modules)


def build_conv3x3_block(in_channels: int, out_channels: int):
    return nn.Sequential(
        helper.conv3x3(in_channels=in_channels, out_channels=out_channels),
        helper.permute_bchw_to_bhwc(),
        helper.ln(in_channels=out_channels),
        helper.permute_bhwc_to_bchw(),
        helper.gelu(),
    )


class PpmBlock(nn.Module):

    def __init__(
        self,
        ppm_scales: Sequence[int],
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        ap_conv_blocks: List[nn.Module] = []
        for ppm_scale in ppm_scales:
            ap_conv_blocks.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(ppm_scale),
                    build_conv1x1_block(in_channels=in_channels, out_channels=out_channels),
                )
            )
        self.ap_conv_blocks = nn.ModuleList(ap_conv_blocks)

        self.final_conv_block = build_conv3x3_block(
            in_channels=in_channels + len(ppm_scales) * out_channels,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        # x: (B, C, H, W)
        shape: Tuple[int, int] = (x.shape[-2], x.shape[-1])
        features = [x]
        for ap_conv_block in self.ap_conv_blocks:
            feature = ap_conv_block(x)
            feature = F.interpolate(feature, size=shape, mode='bilinear')
            features.append(feature)

        features_cat = torch.cat(features, dim=1)
        output = self.final_conv_block(features_cat)
        return output


class UperNextNeck(nn.Module):

    @classmethod
    def build_step1_conv_blocks(
        cls,
        in_channels_group: Sequence[int],
        ppm_scales: Sequence[int],
        inner_channels: int,
    ):
        step1_conv_blocks: List[nn.Module] = []

        # First to second to the last layer, conv1x1 block.
        for in_channels in in_channels_group[:-1]:
            step1_conv_blocks.append(
                build_conv1x1_block(
                    in_channels=in_channels,
                    out_channels=inner_channels,
                )
            )

        # Last layer, PPM block.
        step1_conv_blocks.append(
            PpmBlock(
                ppm_scales=ppm_scales,
                in_channels=in_channels_group[-1],
                out_channels=inner_channels,
            )
        )

        return nn.ModuleList(step1_conv_blocks)

    @classmethod
    def build_step2_conv_blocks(
        cls,
        num_step1_conv_blocks: int,
        inner_channels: int,
    ):
        step2_conv_blocks: List[nn.Module] = []
        # Skip the last layer since it's already been applied conv3x3.
        for _ in range(num_step1_conv_blocks - 1):
            step2_conv_blocks.append(
                build_conv3x3_block(
                    in_channels=inner_channels,
                    out_channels=inner_channels,
                )
            )
        return nn.ModuleList(step2_conv_blocks)

    def __init__(
        self,
        in_channels_group: Sequence[int],
        out_channels: int,
        ppm_scales: Sequence[int] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()

        assert len(in_channels_group) > 1
        assert out_channels % len(in_channels_group) == 0
        inner_channels = out_channels // len(in_channels_group)

        self.step1_conv_blocks = self.build_step1_conv_blocks(
            in_channels_group=in_channels_group,
            ppm_scales=ppm_scales,
            inner_channels=inner_channels,
        )
        self.step2_conv_blocks = self.build_step2_conv_blocks(
            num_step1_conv_blocks=len(self.step1_conv_blocks),
            inner_channels=inner_channels,
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:  # type: ignore
        num_features = len(features)
        assert num_features == len(self.step1_conv_blocks)

        # Step 1.
        outputs = [
            step1_conv_block(features[feature_idx])
            for feature_idx, step1_conv_block in enumerate(self.step1_conv_blocks)
        ]

        # Upsampling & add to the previous layer.
        for feature_idx in range(num_features - 1, 0, -1):
            prev_feature_idx = feature_idx - 1
            prev_feature = outputs[prev_feature_idx]
            prev_shape = (prev_feature.shape[-2], prev_feature.shape[-1])
            outputs[prev_feature_idx] += F.interpolate(
                outputs[feature_idx],
                size=prev_shape,
                mode='bilinear',
            )

        # Step 2.
        for feature_idx, step2_conv_block in enumerate(self.step2_conv_blocks):
            outputs[feature_idx] = step2_conv_block(outputs[feature_idx])

        # Final.
        feature0_shape: Tuple[int, int] = (features[0].shape[-2], features[0].shape[-1])
        for feature_idx in range(1, num_features):
            outputs[feature_idx] = F.interpolate(
                outputs[feature_idx],
                size=feature0_shape,
                mode='bilinear',
            )
        # (B, out_channels, H, W)
        outputs_cat = torch.cat(outputs, dim=1)
        return outputs_cat


class UperNextHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsampling_factor: int = 1,
        init_output_bias: float = 0.0,
    ):
        super().__init__()

        self.upsampling_factor = upsampling_factor

        inner_channels = (in_channels + out_channels) // 2
        self.step1_conv3x3 = build_conv3x3_block(
            in_channels=in_channels,
            out_channels=inner_channels,
        )
        self.step2_conv1x1 = nn.Sequential(
            helper.permute_bchw_to_bhwc(),
            helper.conv1x1(in_channels=inner_channels, out_channels=out_channels),
            helper.permute_bhwc_to_bchw(),
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.constant_(self.step2_conv1x1[1].bias, init_output_bias)  # type: ignore

    def forward(self, fpn_neck_feature: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = fpn_neck_feature

        if self.upsampling_factor > 1:
            x = F.interpolate(
                x,
                size=(
                    x.shape[-2] * self.upsampling_factor,
                    x.shape[-1] * self.upsampling_factor,
                ),
                mode='bilinear',
            )

        x = self.step1_conv3x3(x)
        x = self.step2_conv1x1(x)
        return x
