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
from typing import Tuple, List, Sequence, Dict
from enum import Enum, unique
import logging

import torch
from torch import nn
import attrs

from . import helper
from .convnext import ConvNext
from .pan import PanNeck, PanHead

logger = logging.getLogger(__name__)


def build_double_conv1x1_block(in_channels: int, out_channels: int, factor: int = 4):
    mid_channels = factor * out_channels
    return nn.Sequential(
        helper.conv1x1(in_channels=in_channels, out_channels=mid_channels, use_conv2d=True),
        helper.gelu(),
        helper.conv1x1(in_channels=mid_channels, out_channels=out_channels, use_conv2d=True),
    )


@unique
class LookAgainSize(Enum):
    SMALL = 'small'


@attrs.define
class LookAgainConfig:
    size: LookAgainSize = LookAgainSize.SMALL
    # Mainly for validation and OTA.
    downsampling_factors: Sequence[int] = [4, 8, 16, 32]
    enable_validate_downsampling_factor: bool = True
    # Logits for [char, seal impression]
    rough_classification_num_classes: int = 2
    rough_char_scale_head_init_output_bias: float = 8.0


class LookAgain(nn.Module):

    def __init__(self, config: LookAgainConfig):
        super().__init__()

        self.config = config
        # For JIT.
        self._jit_enable_validate_downsampling_factor = config.enable_validate_downsampling_factor
        self._jit_downsampling_factors = config.downsampling_factors

        if config.size == LookAgainSize.SMALL:
            backbone_creator = ConvNext.create_small
            head_in_channels = 256
        else:
            raise NotImplementedError()

        # Shared backbone, generates 4 feature maps: 1/4, 1/8, 1/16, and 1/32.
        self.backbone = backbone_creator()
        assert len(self.backbone.in_channels_group) == len(config.downsampling_factors)

        # Shared neck.
        self.neck = PanNeck(in_channels_group=self.backbone.in_channels_group)

        # Rough looking.
        self.rough_stem = build_double_conv1x1_block(
            # Use only the 1/4 level.
            in_channels=self.neck.in_channels_group[0],
            out_channels=head_in_channels,
        )
        self.rough_classification_head = PanHead(
            in_channels=head_in_channels,
            out_channels=config.rough_classification_num_classes,
        )
        self.rough_char_scale_head = nn.Sequential(
            PanHead(
                in_channels=head_in_channels,
                out_channels=1,
                init_output_bias=config.rough_char_scale_head_init_output_bias,
            ),
            # Force predicting positive value.
            nn.Softplus(),
        )

        # Precise looking.
        self.precise_stems = nn.ModuleList([
            build_double_conv1x1_block(in_channels=in_channels, out_channels=head_in_channels)
            # Use the 1/8, 1/16, and 1/32 levels.
            for in_channels in self.neck.in_channels_group[1:]
        ])
        self.precise_char_localization_heads = nn.ModuleList([
            # (y, x, h, w).
            PanHead(in_channels=head_in_channels, out_channels=4) for _ in self.precise_stems
        ])
        self.precise_char_objectness_heads = nn.ModuleList([
            PanHead(in_channels=head_in_channels, out_channels=1) for _ in self.precise_stems
        ])
        self.precise_char_orientation_heads = nn.ModuleList([
            PanHead(in_channels=head_in_channels, out_channels=4) for _ in self.precise_stems
        ])

    @classmethod
    def validate_downsampling_factor(
        cls,
        original_tensor: torch.Tensor,
        downsampled_tensor: torch.Tensor,
        downsampling_factor: int,
    ):
        original_height: int = original_tensor.shape[2]
        original_width: int = original_tensor.shape[3]
        downsampled_height: int = downsampled_tensor.shape[2]
        downsampled_width: int = downsampled_tensor.shape[3]
        assert downsampled_height * downsampling_factor == original_height
        assert downsampled_width * downsampling_factor == original_width

    @torch.jit.export  # type: ignore
    def forward_rough(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)

        # Use only the 1/4 level.
        rough_stem_feature = self.rough_stem(neck_features[0])
        del backbone_features, neck_features

        if self._jit_enable_validate_downsampling_factor:
            self.validate_downsampling_factor(
                original_tensor=x,
                downsampled_tensor=rough_stem_feature,
                downsampling_factor=self._jit_downsampling_factors[0],
            )

        rough_classification_logits = self.rough_classification_head(rough_stem_feature)
        rough_char_scale_logits = self.rough_char_scale_head(rough_stem_feature)

        return rough_classification_logits, rough_char_scale_logits

    # NOTE: Can't use Sequence here.
    @torch.jit.export  # type: ignore
    def forward_precise(
        self,
        x: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)

        precise_char_localization_logits_group: List[torch.Tensor] = []
        precise_char_objectness_logits_group: List[torch.Tensor] = []
        precise_char_orientation_logits_group: List[torch.Tensor] = []

        # NOTE: This for loop is design for JIT.
        for (
            neck_feature_idx,
            (
                precise_stem,
                precise_char_localization_head,
                precise_char_objectness_head,
                precise_char_orientation_head,
            ),
        ) in enumerate(
            zip(
                self.precise_stems,
                self.precise_char_localization_heads,
                self.precise_char_objectness_heads,
                self.precise_char_orientation_heads,
            ),
            start=1,
        ):
            # Use the 1/8, 1/16, and 1/32 levels.
            precise_stem_feature = precise_stem(neck_features[neck_feature_idx])

            if self._jit_enable_validate_downsampling_factor:
                self.validate_downsampling_factor(
                    original_tensor=x,
                    downsampled_tensor=precise_stem_feature,
                    downsampling_factor=self._jit_downsampling_factors[neck_feature_idx],
                )

            precise_char_localization_logits_group.append(
                precise_char_localization_head(precise_stem_feature)
            )
            precise_char_objectness_logits_group.append(
                precise_char_objectness_head(precise_stem_feature)
            )
            precise_char_orientation_logits_group.append(
                precise_char_orientation_head(precise_stem_feature)
            )

        return (
            precise_char_localization_logits_group,
            precise_char_objectness_logits_group,
            precise_char_orientation_logits_group,
        )


class LookAgainPostProcessing:

    def __init__(self, downsampling_factors: Sequence[int]):
        self.downsampling_factors = downsampling_factors
        # int -> (1, H, W, 2)
        self.downsampling_factor_to_char_localization_offset: Dict[int, torch.Tensor] = {}

    def get_char_localization_offset(
        self,
        downsampling_factor: int,
        precise_char_localization_logits: torch.Tensor,
    ):
        df_to_offset = self.downsampling_factor_to_char_localization_offset

        target_height, target_width = precise_char_localization_logits.shape[-2:]
        if downsampling_factor not in df_to_offset \
                or df_to_offset[downsampling_factor].shape != (1, target_height, target_width):
            # (H, W)
            offset_y, offset_x = torch.meshgrid(
                torch.arange(target_height),
                torch.arange(target_width),
                indexing='ij',
            )
            # (H, W, 2), offset[y][x] -> (y, x)
            offset = torch.dstack((offset_y, offset_x))
            # (1, H, W, 2)
            offset = offset.unsqueeze(0)
            # Update.
            df_to_offset[downsampling_factor] = offset

        target_device = precise_char_localization_logits.device
        if df_to_offset[downsampling_factor].device != target_device:
            # Udpate device.
            offset = df_to_offset[downsampling_factor]
            df_to_offset[downsampling_factor] = offset.to(target_device)

        return df_to_offset[downsampling_factor]

    def transform_precise_outputs(
        self,
        precise_char_localization_logits_group: Sequence[torch.Tensor],
        precise_char_objectness_logits_group: Sequence[torch.Tensor],
        precise_char_orientation_logits_group: Sequence[torch.Tensor],
    ):
        assert len(self.downsampling_factors) \
            == len(precise_char_localization_logits_group) \
            == len(precise_char_objectness_logits_group) \
            == len(precise_char_orientation_logits_group)

        # For i = 1 to 3.
        # [ (B, Ai, 4), ... ]
        precise_char_localization_bounding_boxes_group: List[torch.Tensor] = []
        # [ (B, Ai, 1), ... ]
        precise_char_objectness_flatten_logits_group: List[torch.Tensor] = []
        precise_char_objectness_probs_group: List[torch.Tensor] = []
        # [ (B, Ai, 4), ... ]
        precise_char_orientation_flatten_logits_group: List[torch.Tensor] = []

        for (
            # D
            downsampling_factor,
            # (B, 4, H, W)
            precise_char_localization_logits,
            # (B, 1, H, W)
            precise_char_objectness_logits,
            # (B, 4, H, W)
            precise_char_orientation_logits,
        ) in zip(
            self.downsampling_factors,
            precise_char_localization_logits_group,
            precise_char_objectness_logits_group,
            precise_char_orientation_logits_group,
        ):
            assert precise_char_localization_logits.shape[-2:] \
                == precise_char_objectness_logits.shape[-2:] \
                == precise_char_orientation_logits.shape[-2:]

            batch_size = precise_char_localization_logits.shape[0]
            assert batch_size \
                == precise_char_objectness_logits.shape[0] \
                == precise_char_orientation_logits.shape[0]

            # 1. Transform localization result.
            # (1, H, W, 2)
            char_localization_offset = self.get_char_localization_offset(
                downsampling_factor,
                precise_char_localization_logits,
            )
            # -> (1, H * W, 2)
            char_localization_offset = char_localization_offset.view(1, -1, 2)

            # -> (B, H, W, 4)
            precise_char_localization_logits = precise_char_localization_logits.permute(0, 2, 3, 1)
            # -> (B, H * W, 4), 4 for (y, x, h, w)
            precise_char_localization_logits = \
                precise_char_localization_logits.view(batch_size, -1, 4)

            # Transform the center point.
            precise_char_localization_logits[..., :2] = \
                (precise_char_localization_logits[..., :2] + char_localization_offset) \
                * downsampling_factor
            # Transform the shape of bounding box,
            precise_char_localization_logits[..., 2:] = \
                torch.exp(precise_char_localization_logits[..., 2:]) * downsampling_factor

            # Keep the transformed bounding boxes.
            precise_char_localization_bounding_boxes_group.append(precise_char_localization_logits)

            # 2. Transform objectness result.
            # -> (B, H, W, 1)
            precise_char_objectness_logits = precise_char_objectness_logits.permute(0, 2, 3, 1)
            # -> (B, H * W, 1)
            precise_char_objectness_logits = \
                precise_char_objectness_logits.view(batch_size, -1, 1)

            # Keep the flatten logits and the transformed probs.
            precise_char_objectness_flatten_logits_group.append(precise_char_objectness_logits)
            precise_char_objectness_probs_group.append(precise_char_objectness_logits.sigmoid())

            # 3. Transform orientation result.
            # -> (B, H, W, 4)
            precise_char_orientation_logits = precise_char_orientation_logits.permute(0, 2, 3, 1)
            # -> (B, H * W, 4), 4 for (up, down, left, right) logits.
            precise_char_orientation_logits = \
                precise_char_orientation_logits.view(batch_size, -1, 4)

            # Keep only the flatten logits.
            precise_char_orientation_flatten_logits_group.append(precise_char_orientation_logits)

        # Combine.
        # A = SUM(Ai)
        # (B, A, 4)
        precise_char_localization_bounding_boxes = torch.cat(
            precise_char_localization_bounding_boxes_group,
            dim=1,
        )
        # (B, A, 1)
        precise_char_objectness_flatten_logits = torch.cat(
            precise_char_objectness_flatten_logits_group,
            dim=1,
        )
        # (B, A, 1)
        precise_char_objectness_probs = torch.cat(
            precise_char_objectness_probs_group,
            dim=1,
        )
        # (B, A, 4)
        precise_char_orientation_flatten_logits = torch.cat(
            precise_char_orientation_flatten_logits_group,
            dim=1,
        )

        return (
            precise_char_localization_bounding_boxes,
            precise_char_objectness_flatten_logits,
            precise_char_objectness_probs,
            precise_char_orientation_flatten_logits,
        )
