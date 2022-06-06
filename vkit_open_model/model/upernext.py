from typing import Sequence, Tuple, List

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


class UperNext(nn.Module):

    def __init__(
        self,
        in_channels_group: Sequence[int],
        mid_channels: int,
        ppm_scales: Sequence[int],
        out_channels: int,
    ) -> None:
        super().__init__()

        assert len(in_channels_group) > 1

        step1_conv_blocks: List[nn.Module] = []
        for in_channels in in_channels_group[:-1]:
            step1_conv_blocks.append(
                build_conv1x1_block(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                )
            )
        self.step1_conv_blocks = nn.ModuleList(step1_conv_blocks)

        self.step1_ppm_block = PpmBlock(
            ppm_scales=ppm_scales,
            in_channels=in_channels_group[-1],
            out_channels=mid_channels,
        )

        step2_conv_blocks: List[nn.Module] = []
        for in_channels in in_channels_group[:-1]:
            step2_conv_blocks.append(
                build_conv3x3_block(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                )
            )
        self.step2_conv_blocks = nn.ModuleList(step2_conv_blocks)

        self.final_conv_block = nn.Sequential(
            build_conv3x3_block(
                in_channels=len(in_channels_group) * mid_channels,
                out_channels=mid_channels,
            ),
            build_conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
            ),
        )

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def create_tiny(out_channels: int):
        return UperNext(
            in_channels_group=(96, 192, 384, 768),
            mid_channels=512,
            ppm_scales=(1, 2, 3, 6),
            out_channels=out_channels,
        )

    @staticmethod
    def create_small(out_channels: int):
        return UperNext(
            in_channels_group=(96, 192, 384, 768),
            mid_channels=512,
            ppm_scales=(1, 2, 3, 6),
            out_channels=out_channels,
        )

    @staticmethod
    def create_base(out_channels: int):
        return UperNext(
            in_channels_group=(192, 384, 768, 1536),
            mid_channels=512,
            ppm_scales=(1, 2, 3, 6),
            out_channels=out_channels,
        )

    @staticmethod
    def create_large(out_channels: int):
        return UperNext(
            in_channels_group=(256, 512, 1024, 2048),
            mid_channels=512,
            ppm_scales=(1, 2, 3, 6),
            out_channels=out_channels,
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:  # type: ignore
        num_features = len(features)
        assert num_features == len(self.step1_conv_blocks) + 1

        # Step 1.
        outputs: List[torch.Tensor] = []
        # Not the last layers.
        for feature_idx, step1_conv_block in enumerate(self.step1_conv_blocks):
            outputs.append(step1_conv_block(features[feature_idx]))
        # Last layer.
        outputs.append(self.step1_ppm_block(features[-1]))

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
        outputs_cat = torch.cat(outputs, dim=1)
        output = self.final_conv_block(outputs_cat)
        return output
