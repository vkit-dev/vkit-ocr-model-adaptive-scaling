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
from typing import Sequence, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from . import helper


def build_dconv3x3_block(in_channels: int):
    return nn.Sequential(
        helper.dconv3x3(in_channels=in_channels),
        helper.permute_bchw_to_bhwc(),
        helper.ln(in_channels=in_channels),
        helper.permute_bhwc_to_bchw(),
    )


def build_double_conv1x1_block(in_channels: int, out_channels: int, factor: int = 4):
    mid_channels = factor * out_channels
    return nn.Sequential(
        helper.conv1x1(in_channels=in_channels, out_channels=mid_channels, use_conv2d=True),
        helper.gelu(),
        helper.conv1x1(in_channels=mid_channels, out_channels=out_channels, use_conv2d=True),
    )


def build_dconv3x3_and_double_conv1x1_block(in_channels: int, factor: int = 4):
    mid_channels = factor * in_channels
    return nn.Sequential(
        helper.dconv3x3(in_channels=in_channels),
        helper.permute_bchw_to_bhwc(),
        helper.ln(in_channels=in_channels),
        helper.conv1x1(in_channels=in_channels, out_channels=mid_channels),
        helper.gelu(),
        helper.conv1x1(in_channels=mid_channels, out_channels=in_channels),
        helper.permute_bhwc_to_bchw(),
    )


class PanNeckTopDownTopBlock(nn.Module):

    def __init__(self, upper_channels: int, lower_channels: int):
        super().__init__()

        self.double_conv1x1 = build_double_conv1x1_block(
            in_channels=upper_channels,
            out_channels=lower_channels,
        )

    def forward(self, backbone_feature: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.double_conv1x1(backbone_feature)


class PanNeckTopDownNormalBlock(nn.Module):

    def __init__(self, upper_channels: int, lower_channels: int):
        super().__init__()

        self.dconv3x3 = build_dconv3x3_block(in_channels=upper_channels)
        self.double_conv1x1 = build_double_conv1x1_block(
            in_channels=2 * upper_channels,
            out_channels=lower_channels,
        )

    def forward(  # type: ignore
        self,
        upper_neck_feature: torch.Tensor,
        backbone_feature: torch.Tensor,
    ) -> torch.Tensor:
        # Upsample.
        upper_neck_feature = F.interpolate(
            upper_neck_feature,
            size=(backbone_feature.shape[-2], backbone_feature.shape[-1]),
            mode='nearest',
        )

        # Apply dconv3x3.
        upper_neck_feature = self.dconv3x3(upper_neck_feature)

        # Concatenate.
        feature = torch.cat((upper_neck_feature, backbone_feature), dim=1)

        # Apply conv1x1 for top-down path.
        lower_neck_feature = self.double_conv1x1(feature)

        return lower_neck_feature


class PanNeckBottomUpBlock(nn.Module):

    def __init__(self, upper_channels: int, lower_channels: int):
        super().__init__()

        self.pconv2x2 = helper.pconv2x2(in_channels=lower_channels, out_channels=lower_channels)
        self.double_conv1x1 = build_double_conv1x1_block(
            in_channels=2 * lower_channels,
            out_channels=upper_channels,
        )

    def forward(  # type: ignore
        self,
        lower_neck_feature: torch.Tensor,
        top_down_hori_feature: torch.Tensor,
    ) -> torch.Tensor:
        # Downsample.
        lower_neck_feature = self.pconv2x2(lower_neck_feature)

        # Concatenate.
        feature = torch.cat((lower_neck_feature, top_down_hori_feature), dim=1)

        # Apply conv1x1.
        return self.double_conv1x1(feature)


class PanNeck(nn.Module):

    @classmethod
    def build_top_down_blocks(cls, in_channels_group: Sequence[int]):
        top_down_top_block: Optional[nn.Module] = None
        top_down_normal_blocks: List[nn.Module] = []

        last_idx = len(in_channels_group) - 1
        for idx in range(last_idx + 1):
            is_top = (idx == last_idx)
            is_bottom = (idx == 0)

            upper_channels = in_channels_group[idx]

            if is_bottom:
                lower_channels = upper_channels
            else:
                lower_channels = in_channels_group[idx - 1]

            if is_top:
                top_down_top_block = PanNeckTopDownTopBlock(
                    upper_channels=upper_channels,
                    lower_channels=lower_channels,
                )
            else:
                top_down_normal_blocks.append(
                    PanNeckTopDownNormalBlock(
                        upper_channels=upper_channels,
                        lower_channels=lower_channels,
                    )
                )

        assert top_down_top_block is not None
        top_down_normal_blocks = list(reversed(top_down_normal_blocks))

        return top_down_top_block, nn.ModuleList(top_down_normal_blocks)

    @classmethod
    def build_bottom_up_blocks(cls, in_channels_group: Sequence[int]):
        bottom_up_blocks: List[nn.Module] = []
        last_idx = len(in_channels_group) - 1
        for idx in range(last_idx):
            upper_channels = in_channels_group[idx + 1]
            lower_channels = in_channels_group[idx]
            bottom_up_blocks.append(
                PanNeckBottomUpBlock(
                    upper_channels=upper_channels,
                    lower_channels=lower_channels,
                )
            )
        return nn.ModuleList(bottom_up_blocks)

    def __init__(self, in_channels_group: Sequence[int]):
        super().__init__()

        self.in_channels_group = in_channels_group

        assert len(in_channels_group) > 1
        (
            self.top_down_top_block,
            self.top_down_normal_blocks,
        ) = self.build_top_down_blocks(in_channels_group)

        self.bottom_up_blocks = self.build_bottom_up_blocks(in_channels_group)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, backbone_features: List[torch.Tensor]) -> List[torch.Tensor]:  # type: ignore
        assert len(backbone_features) - 1 == len(self.top_down_normal_blocks)

        # Top-down path.
        top_down_hori_features: List[torch.Tensor] = []

        upper_neck_feature = self.top_down_top_block(backbone_feature=backbone_features[-1])
        top_down_hori_features.append(upper_neck_feature)

        for top_down_normal_block_idx, top_down_normal_block in \
                enumerate(self.top_down_normal_blocks, start=2):
            upper_neck_feature = top_down_normal_block(
                upper_neck_feature=upper_neck_feature,
                backbone_feature=backbone_features[-top_down_normal_block_idx],
            )
            top_down_hori_features.append(upper_neck_feature)

        assert len(top_down_hori_features) == len(backbone_features)
        assert len(top_down_hori_features) - 1 == len(self.bottom_up_blocks)

        # Bottom-up path.
        output_features: List[torch.Tensor] = []

        lower_neck_feature = top_down_hori_features.pop()
        output_features.append(lower_neck_feature)

        for bottom_up_block in self.bottom_up_blocks:
            top_down_hori_feature = top_down_hori_features.pop()
            lower_neck_feature = bottom_up_block(
                lower_neck_feature=lower_neck_feature,
                top_down_hori_feature=top_down_hori_feature,
            )
            output_features.append(lower_neck_feature)

        return output_features


class PanHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_output_bias: float = 0.0,
    ):
        super().__init__()

        self.step1_conv = nn.Sequential(
            build_dconv3x3_and_double_conv1x1_block(in_channels=in_channels),
            build_dconv3x3_and_double_conv1x1_block(in_channels=in_channels),
        )
        self.step2_conv = build_double_conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            factor=8,
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.constant_(self.step2_conv[0].bias, 0.0)  # type: ignore
        nn.init.constant_(self.step2_conv[2].bias, init_output_bias)  # type: ignore

    def forward(self, neck_feature: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = neck_feature
        x = self.step1_conv(x)
        x = self.step2_conv(x)
        return x
