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
from typing import Sequence, List

import torch
from torch import nn
from torch.nn import functional as F

from . import helper


def build_conv1x1_block(in_channels: int, out_channels: int):
    return nn.Sequential(
        helper.permute_bchw_to_bhwc(),
        helper.conv1x1(in_channels=in_channels, out_channels=out_channels),
        helper.ln(in_channels=out_channels),
        helper.permute_bhwc_to_bchw(),
        helper.gelu(),
    )


def build_conv3x3_block(in_channels: int, out_channels: int):
    return nn.Sequential(
        helper.conv3x3(in_channels=in_channels, out_channels=out_channels),
        helper.permute_bchw_to_bhwc(),
        helper.ln(in_channels=out_channels),
        helper.permute_bhwc_to_bchw(),
        helper.gelu(),
    )


def build_pconv2x2_block(in_channels: int, out_channels: int):
    return nn.Sequential(
        helper.pconv2x2(in_channels=in_channels, out_channels=out_channels),
        helper.permute_bchw_to_bhwc(),
        helper.ln(in_channels=out_channels),
        helper.permute_bhwc_to_bchw(),
        helper.gelu(),
    )


class PanNeck(nn.Module):

    @classmethod
    def build_step1_conv_blocks(
        cls,
        in_channels_group: Sequence[int],
        out_channels: int,
    ):
        step1_conv_blocks: List[nn.Module] = []
        for in_channels in in_channels_group:
            step1_conv_blocks.append(
                build_conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                )
            )
        return nn.ModuleList(step1_conv_blocks)

    @classmethod
    def build_step2_conv_blocks(
        cls,
        in_channels_group: Sequence[int],
        out_channels: int,
    ):
        step2_conv_blocks: List[nn.Module] = []
        for _ in in_channels_group:
            step2_conv_blocks.append(
                build_conv3x3_block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                )
            )
        return nn.ModuleList(step2_conv_blocks)

    @classmethod
    def build_step3_conv_blocks(
        cls,
        in_channels_group: Sequence[int],
        out_channels: int,
    ):
        step3_conv_blocks: List[nn.Module] = []
        for idx, _ in enumerate(in_channels_group):
            if idx >= len(in_channels_group) - 1:
                break
            step3_conv_blocks.append(
                build_pconv2x2_block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                )
            )
        return nn.ModuleList(step3_conv_blocks)

    @classmethod
    def build_step4_conv_blocks(
        cls,
        in_channels_group: Sequence[int],
        out_channels: int,
    ):
        step4_conv_blocks: List[nn.Module] = []
        for idx, _ in enumerate(in_channels_group):
            if idx == 0:
                continue
            step4_conv_blocks.append(
                build_conv3x3_block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                )
            )
        return nn.ModuleList(step4_conv_blocks)

    def __init__(
        self,
        in_channels_group: Sequence[int],
        out_channels: int,
    ) -> None:
        super().__init__()

        assert len(in_channels_group) > 1

        # Top-down.
        self.step1_conv_blocks = self.build_step1_conv_blocks(
            in_channels_group=in_channels_group,
            out_channels=out_channels,
        )
        self.step2_conv_blocks = self.build_step2_conv_blocks(
            in_channels_group=in_channels_group,
            out_channels=out_channels,
        )
        assert len(self.step1_conv_blocks) == len(self.step2_conv_blocks)

        # Bottom-up.
        self.step3_conv_blocks = self.build_step3_conv_blocks(
            in_channels_group=in_channels_group,
            out_channels=out_channels,
        )
        self.step4_conv_blocks = self.build_step4_conv_blocks(
            in_channels_group=in_channels_group,
            out_channels=out_channels,
        )
        assert len(self.step3_conv_blocks) == len(self.step4_conv_blocks)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:  # type: ignore
        num_features = len(features)
        assert num_features == len(self.step1_conv_blocks)

        # Step 1. (top-down)
        outputs = [
            step1_conv_block(features[feature_idx])
            for feature_idx, step1_conv_block in enumerate(self.step1_conv_blocks)
        ]

        # Upsampling & add to the previous layer.
        for feature_idx in range(num_features - 1, 0, -1):
            prev_feature_idx = feature_idx - 1
            feature = outputs[feature_idx]
            height = feature.shape[-2]
            width = feature.shape[-1]
            outputs[prev_feature_idx] += F.interpolate(
                outputs[feature_idx],
                size=(height * 2, width * 2),
                mode='nearest',
            )

        # Step 2. (top-down)
        for feature_idx, step2_conv_block in enumerate(self.step2_conv_blocks):
            outputs[feature_idx] = step2_conv_block(outputs[feature_idx])

        # Step 3. (bottom-up)
        for feature_idx, step3_conv_block in enumerate(self.step3_conv_blocks):
            next_feature_idx = feature_idx + 1
            outputs[next_feature_idx] += step3_conv_block(outputs[feature_idx])

        # Step 4. (bottom-up)
        for feature_idx, step4_conv_block in enumerate(self.step4_conv_blocks, start=1):
            outputs[feature_idx] = step4_conv_block(outputs[feature_idx])

        return outputs
