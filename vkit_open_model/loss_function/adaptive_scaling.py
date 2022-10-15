from typing import Tuple

import torch
from vkit.element import Box
import attrs

from .weighted_bce_with_logits import WeightedBceWithLogitsLossFunction
from .focal_with_logits import FocalWithLogitsLossFunction
from .dice import DiceLossFunction
from .l1 import L1LossFunction
from .cross_entropy_with_logits import CrossEntropyWithLogitsLossFunction


@attrs.define
class AdaptiveScalingRoughLossFunctionConifg:
    bce_negative_ratio: float = 3.0
    bce_factor: float = 0.0
    focal_factor: float = 5.0
    dice_factor: float = 1.0
    l1_factor: float = 1.0
    downsampled_score_map_min: float = 1.1
    scale_feature_min: float = 1.1


class AdaptiveScalingRoughLossFunction:

    def __init__(self, config: AdaptiveScalingRoughLossFunctionConifg):
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
        # Model predictions.
        # (B, 1, H, W)
        rough_char_mask_feature: torch.Tensor,
        rough_char_scale_feature: torch.Tensor,
        # Ground truths.
        # (B, CH, CW)
        downsampled_mask: torch.Tensor,
        downsampled_score_map: torch.Tensor,
        downsampled_shape: Tuple[int, int],
        downsampled_core_box: Box,
    ) -> torch.Tensor:
        # (B, 1, H, W)
        assert rough_char_mask_feature.shape == rough_char_scale_feature.shape
        assert rough_char_mask_feature.shape[1:] == (1, *downsampled_shape)

        # (B, H, W)
        rough_char_mask_feature = torch.squeeze(rough_char_mask_feature, dim=1)
        rough_char_scale_feature = torch.squeeze(rough_char_scale_feature, dim=1)

        # (B, CH, CW)
        dc_box = downsampled_core_box

        rough_char_mask_feature = rough_char_mask_feature[
            :,
            dc_box.up:dc_box.down + 1,
            dc_box.left:dc_box.right + 1
        ]  # yapf: disable
        rough_char_scale_feature = rough_char_scale_feature[
            :,
            dc_box.up:dc_box.down + 1,
            dc_box.left:dc_box.right + 1
        ]  # yapf: disable

        loss = 0.0

        # Mask.
        if self.config.bce_factor > 0.0:
            loss += self.config.bce_factor * self.weighted_bce_with_logits(
                pred=rough_char_mask_feature,
                gt=downsampled_mask,
            )

        if self.config.focal_factor > 0.0:
            loss += self.config.focal_factor * self.focal_with_logits(
                pred=rough_char_mask_feature,
                gt=downsampled_mask,
            )

        if self.config.dice_factor > 0.0:
            loss += self.config.dice_factor * self.dice(
                pred=torch.sigmoid(rough_char_mask_feature),
                gt=downsampled_mask,
            )

        # Scale.
        if self.config.l1_factor > 0.0:
            # NOTE: critical mask!
            l1_mask = ((rough_char_scale_feature > self.config.scale_feature_min)
                       & (downsampled_score_map > self.config.downsampled_score_map_min)
                       & downsampled_mask.bool()).float()
            rough_char_scale_feature = torch.clamp(
                rough_char_scale_feature,
                min=self.config.scale_feature_min,
            )
            downsampled_score_map = torch.clamp(
                downsampled_score_map,
                min=self.config.downsampled_score_map_min,
            )
            # Log space + smooth L1 to model the relative scale difference.
            loss += self.config.l1_factor * self.l1(
                pred=torch.log(rough_char_scale_feature),
                gt=torch.log(downsampled_score_map),
                mask=l1_mask,
            )

        assert not isinstance(loss, float)
        return loss


@attrs.define
class AdaptiveScalingPreciseLossFunctionConifg:
    char_prob_l1_factor: float = 5.0
    char_up_left_offset_l1_factor: float = 1.0
    char_corner_angle_cross_entropy_factor: float = 5.0
    char_corner_distance_l1_factor: float = 1.0


class AdaptiveScalingPreciseLossFunction:

    def __init__(self, config: AdaptiveScalingPreciseLossFunctionConifg):
        self.config = config

        # Prob.
        self.char_prob_l1 = L1LossFunction(smooth=True, smooth_beta=0.5)
        # Up-left corner.
        self.char_up_left_offset_l1 = L1LossFunction(smooth=True, smooth_beta=5.0)
        # Corner angle.
        self.char_corner_angle_cross_entropy = CrossEntropyWithLogitsLossFunction()
        # Corner distance.
        self.char_corner_distance_l1 = L1LossFunction(smooth=True, smooth_beta=5.0)

    @classmethod
    def get_label_point_feature(
        cls,
        # (B, *, H, W)
        feature: torch.Tensor,
        # (B, P)
        label_point_y: torch.Tensor,
        label_point_x: torch.Tensor,
    ):
        batch_size = feature.shape[0]
        assert batch_size == label_point_y.shape[0] == label_point_x.shape[0]
        # (B, P, *)
        return feature[torch.arange(batch_size)[:, None], :, label_point_y, label_point_x]

    def __call__(
        self,
        # Model predictions.
        # (B, 1, H, W)
        precise_char_prob_feature: torch.Tensor,
        # (B, 2, H, W)
        precise_char_up_left_corner_offset_feature: torch.Tensor,
        # (B, 4, H, W)
        precise_char_corner_angle_feature: torch.Tensor,
        # (B, 3, H, W)
        precise_char_corner_distance_feature: torch.Tensor,
        # Ground truths.
        # (B, CH, CW)
        downsampled_char_prob_score_map: torch.Tensor,
        downsampled_char_mask: torch.Tensor,
        downsampled_shape: Tuple[int, int],
        downsampled_core_box: Box,
        # Label points.
        # (B, P)
        downsampled_label_point_y: torch.Tensor,
        downsampled_label_point_x: torch.Tensor,
        # (B, P, 2)
        char_up_left_offsets: torch.Tensor,
        # (B, P, 4)
        char_corner_angles: torch.Tensor,
        # (B, P, 3)
        char_corner_distances: torch.Tensor,
    ) -> torch.Tensor:
        # Prob.
        # (B, H, W)
        assert precise_char_prob_feature.shape[1:] == (1, *downsampled_shape)
        precise_char_prob_feature = torch.squeeze(precise_char_prob_feature, dim=1)

        # (B, CH, CW)
        precise_char_prob_feature = precise_char_prob_feature[
            :,
            downsampled_core_box.up:downsampled_core_box.down + 1,
            downsampled_core_box.left:downsampled_core_box.right + 1
        ]  # yapf: disable

        # Up-left corner.
        # (B, P, 2)
        precise_char_up_left_corner_offset_label_point_feature = self.get_label_point_feature(
            feature=precise_char_up_left_corner_offset_feature,
            label_point_y=downsampled_label_point_y,
            label_point_x=downsampled_label_point_x,
        )

        # Corner angle.
        # (B, P, 4)
        precise_char_corner_angle_label_point_feature = self.get_label_point_feature(
            feature=precise_char_corner_angle_feature,
            label_point_y=downsampled_label_point_y,
            label_point_x=downsampled_label_point_x,
        )
        # (B, 4, P), required by torch.nn.functional.cross_entropy.
        precise_char_corner_angle_label_point_feature = \
            precise_char_corner_angle_label_point_feature.transpose(1, 2)
        char_corner_angles = char_corner_angles.transpose(1, 2)

        # Corner distance.
        # (B, P, 3)
        precise_char_corner_distance_label_point_feature = self.get_label_point_feature(
            feature=precise_char_corner_distance_feature,
            label_point_y=downsampled_label_point_y,
            label_point_x=downsampled_label_point_x,
        )

        loss = 0.0

        if self.config.char_prob_l1_factor > 0:
            loss += self.config.char_prob_l1_factor * self.char_prob_l1(
                pred=torch.sigmoid(precise_char_prob_feature),
                gt=downsampled_char_prob_score_map,
                mask=downsampled_char_mask,
            )

        if self.config.char_up_left_offset_l1_factor > 0:
            loss += self.config.char_up_left_offset_l1_factor * self.char_up_left_offset_l1(
                pred=precise_char_up_left_corner_offset_label_point_feature,
                gt=char_up_left_offsets,
            )

        if self.config.char_corner_angle_cross_entropy_factor > 0:
            factor = self.config.char_corner_angle_cross_entropy_factor
            loss += factor * self.char_corner_angle_cross_entropy(
                pred=precise_char_corner_angle_label_point_feature,
                gt=char_corner_angles,
            )

        if self.config.char_corner_distance_l1_factor > 0:
            loss += self.config.char_corner_distance_l1_factor * self.char_corner_distance_l1(
                pred=precise_char_corner_distance_label_point_feature,
                gt=char_corner_distances,
            )

        assert not isinstance(loss, float)
        return loss
