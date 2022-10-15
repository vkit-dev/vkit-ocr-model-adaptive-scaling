from typing import Tuple, List, Sequence
import math
import logging

import torch
import numpy as np
import cv2 as cv
import iolite as io
import attrs

from vkit.utility import PathType
from vkit.element import Polygon, Mask, ScoreMap, Image
from vkit.pipeline.text_detection.page_text_region import (
    stack_flattened_text_regions,
    FlattenedTextRegion,
    TextRegionFlattener,
)

from .opt import pad_mat_to_make_divisible

logger = logging.getLogger(__name__)


@attrs.define
class AdaptiveScalingInferencingConfig:
    model_jit: PathType
    device: str = 'cpu'
    backbone_downsampling_factor: int = 32
    rough_downsample_short_side_legnth: int = 720
    rough_char_mask_positive_thr: float = 0.5
    rough_valid_char_height_min: float = 3.0
    precise_text_region_flattener_typical_long_side_ratio_min: float = 3.0
    precise_text_region_flattener_text_region_polygon_dilate_ratio: float = 0.8
    precise_flattened_text_region_resized_char_height_median: int = 36
    precise_flattened_text_region_resized_ratio_min: float = 0.25
    precise_stack_flattened_text_regions_page_pad: int = 10
    precise_stack_flattened_text_regions_pad: int = 2


@attrs.define
class AdaptiveScalingInferencingRoughInferResult:
    resized_shape: Tuple[int, int]
    padded_image: Image
    rough_char_mask: Mask
    rough_char_scale_score_map: ScoreMap


@attrs.define
class AdaptiveScalingInferencingPresiceInferResult:
    padded_image: Image
    precise_char_prob_score_map: ScoreMap
    precise_char_up_left_corner_offset_score_map: ScoreMap
    precise_char_corner_angle_score_map: ScoreMap
    precise_char_corner_distance_score_map: ScoreMap


class AdaptiveScalingInferencing:

    def __init__(self, config: AdaptiveScalingInferencingConfig):
        self.config = config

        model_jit = torch.jit.load(  # type: ignore
            io.folder(self.config.model_jit, expandvars=True),
            map_location=self.config.device,
        )
        model_jit.eval()
        self.model_jit: torch.jit.ScriptModule = model_jit  # type: ignore

    def rough_infer(self, image: Image):
        image = image.to_rgb_image()

        # Downsample image short-side if needed.
        if min(image.height, image.width) > self.config.rough_downsample_short_side_legnth:
            if image.height < image.width:
                resized_image = image.to_resized_image(
                    resized_height=self.config.rough_downsample_short_side_legnth,
                    cv_resize_interpolation=cv.INTER_AREA,
                )
            else:
                resized_image = image.to_resized_image(
                    resized_width=self.config.rough_downsample_short_side_legnth,
                    cv_resize_interpolation=cv.INTER_AREA,
                )
            image = resized_image

        # Pad image.mat for backbone.
        image_mat = pad_mat_to_make_divisible(
            image.mat,
            downsampling_factor=self.config.backbone_downsampling_factor,
        )
        padded_image = Image(mat=image_mat)

        # (H, W, 3) -> (3, H, W)
        image_mat = np.transpose(image_mat, axes=(2, 0, 1)).astype(np.float32)
        # (3, H, W) -> (1, 3, H, W)
        x = torch.from_numpy(image_mat).unsqueeze(0)
        # Device.
        x = x.to(self.config.device, non_blocking=True)

        # Feed to model.
        with torch.no_grad():
            # (1, 1, H / 2, D / 2)
            (
                rough_char_mask_feature,
                rough_char_scale_feature,
            ) = self.model_jit.forward_rough(x)  # type: ignore

        # (1, 1, H / 2, D / 2)) -> (H / 2, D / 2))
        rough_char_mask_feature = rough_char_mask_feature[0][0]
        rough_char_scale_feature = rough_char_scale_feature[0][0]
        # Assert shape.
        assert rough_char_mask_feature.shape == rough_char_scale_feature.shape
        assert rough_char_mask_feature.shape[0] == padded_image.height // 2
        assert rough_char_mask_feature.shape[1] == padded_image.width // 2

        # Generate rough_char_mask & rough_char_scale_score_map.
        rough_char_mask_feature = torch.sigmoid_(rough_char_mask_feature)
        rough_char_mask_feature = torch.greater_equal(
            rough_char_mask_feature,
            self.config.rough_char_mask_positive_thr,
        )
        rough_char_mask_mat = rough_char_mask_feature.numpy().astype(np.uint8)

        rough_char_scale_score_map_mat = rough_char_scale_feature.numpy().astype(np.float32)

        # Force padding to be negative.
        if image.height < padded_image.height:
            pad_vert_begin = math.ceil(image.height / 2)
            if pad_vert_begin < rough_char_mask_mat.shape[0]:
                rough_char_mask_mat[pad_vert_begin:] = 0
                rough_char_scale_score_map_mat[pad_vert_begin:] = 0.0

        if image.width < padded_image.width:
            pad_hori_begin = math.ceil(image.width / 2)
            if pad_hori_begin < rough_char_mask_mat.shape[1]:
                rough_char_mask_mat[:, pad_hori_begin:] = 0
                rough_char_scale_score_map_mat[:, pad_hori_begin:] = 0.0

        # Clear char height that is too small.
        np_invalid_mask = rough_char_scale_score_map_mat < self.config.rough_valid_char_height_min
        rough_char_scale_score_map_mat[np_invalid_mask] = 0.0

        rough_char_mask = Mask(mat=rough_char_mask_mat)
        rough_char_scale_score_map = ScoreMap(
            mat=rough_char_scale_score_map_mat,
            is_prob=False,
        )

        # E2E 2x downsampling.
        resized_shape = (
            math.ceil(image.height / 2),
            math.ceil(image.width / 2),
        )

        return AdaptiveScalingInferencingRoughInferResult(
            resized_shape=resized_shape,
            padded_image=padded_image,
            rough_char_mask=rough_char_mask,
            rough_char_scale_score_map=rough_char_scale_score_map,
        )

    def build_flattened_text_regions(
        self,
        image: Image,
        rough_infer_result: AdaptiveScalingInferencingRoughInferResult,
    ):
        resized_shape = rough_infer_result.resized_shape
        resized_height, _ = resized_shape
        rough_char_mask = rough_infer_result.rough_char_mask
        rough_char_scale_score_map = rough_infer_result.rough_char_scale_score_map

        # Build disconnected polygons from char mask.
        rough_polygons = rough_char_mask.to_disconnected_polygons()

        # Build text_region_polygons.
        #
        # NOTE: The orginal image is
        # 1. probably resized,
        # 2. then probably padded,
        # 3. and finally downsampled 2X by backbone.
        #
        # Resize the rough_polygon back is fine.
        text_region_polygons: List[Polygon] = []
        for rough_polygon in rough_polygons:
            text_region_polygons.append(
                rough_polygon.to_conducted_resized_polygon(
                    resized_shape,
                    resized_height=image.height,
                    resized_width=image.width,
                )
            )

        # Flatten text regions.
        typical_long_side_ratio_min = \
            self.config.precise_text_region_flattener_typical_long_side_ratio_min
        text_region_polygon_dilate_ratio = \
            self.config.precise_text_region_flattener_text_region_polygon_dilate_ratio
        text_region_flattener = TextRegionFlattener(
            typical_long_side_ratio_min=typical_long_side_ratio_min,
            text_region_polygon_dilate_ratio=text_region_polygon_dilate_ratio,
            image=image,
            text_region_polygons=text_region_polygons,
        )
        flattened_text_regions = text_region_flattener.flattened_text_regions
        assert len(text_region_polygons) == len(flattened_text_regions)

        # Collect the median of char heights.
        # NOTE: The predicted char height is based on resized image.
        inverse_resized_ratio = image.height / (resized_height * 2)
        char_height_medians: List[float] = []
        for rough_polygon in rough_polygons:
            char_height_score_map = rough_polygon.extract_score_map(rough_char_scale_score_map)
            np_mask = (char_height_score_map.mat > 0)
            if not np_mask.any():
                char_height_medians.append(0.0)
            else:
                char_height_medians.append(
                    float(np.median(char_height_score_map.mat[np_mask])) * inverse_resized_ratio
                )
        assert len(text_region_polygons) == len(char_height_medians)

        # Resize.
        resized_char_height_median = \
            self.config.precise_flattened_text_region_resized_char_height_median
        resized_ratio_min = self.config.precise_flattened_text_region_resized_ratio_min
        resized_side_min = round(resized_char_height_median * resized_ratio_min)

        resized_flattened_text_regions: List[FlattenedTextRegion] = []
        for flattened_text_region, char_height_median in zip(
            flattened_text_regions, char_height_medians
        ):
            if char_height_median <= 0.0:
                logger.warning('invalid char_height_median, skip')
                continue

            scale = resized_char_height_median / char_height_median
            resized_height = round(flattened_text_region.height * scale)
            resized_width = round(flattened_text_region.width * scale)

            if resized_height < resized_side_min and resized_width < resized_side_min:
                logger.warning('resized_height and resized_width too small, skip')
                continue

            resized_flattened_text_regions.append(
                flattened_text_region.to_resized_flattened_text_region(
                    resized_height=resized_height,
                    resized_width=resized_width,
                )
            )

        return resized_flattened_text_regions

    def stack_flattened_text_regions(self, flattened_text_regions: Sequence[FlattenedTextRegion]):
        image, boxes, _ = stack_flattened_text_regions(
            page_pad=self.config.precise_stack_flattened_text_regions_page_pad,
            flattened_text_regions_pad=self.config.precise_stack_flattened_text_regions_pad,
            flattened_text_regions=flattened_text_regions,
        )
        return image, boxes

    def precise_infer(self, image: Image):
        # Pad image.mat for backbone.
        image_mat = pad_mat_to_make_divisible(
            image.mat,
            downsampling_factor=self.config.backbone_downsampling_factor,
        )
        padded_image = Image(mat=image_mat)

        # (H, W, 3) -> (3, H, W)
        image_mat = np.transpose(image_mat, axes=(2, 0, 1)).astype(np.float32)
        # (3, H, W) -> (1, 3, H, W)
        x = torch.from_numpy(image_mat).unsqueeze(0)
        # Device.
        x = x.to(self.config.device, non_blocking=True)

        # Feed to model.
        with torch.no_grad():
            # (1, 1, H / 2, D / 2)
            (
                precise_char_prob_feature,
                precise_char_up_left_corner_offset_feature,
                precise_char_corner_angle_feature,
                precise_char_corner_distance_feature,
            ) = self.model_jit.forward_precise(x)  # type: ignore
        # (1, 1, H / 2, D / 2)) -> (H / 2, D / 2))
        precise_char_prob_feature = precise_char_prob_feature[0][0]
        precise_char_up_left_corner_offset_feature = \
            precise_char_up_left_corner_offset_feature[0][0]
        precise_char_corner_angle_feature = precise_char_corner_angle_feature[0][0]
        precise_char_corner_distance_feature = precise_char_corner_distance_feature[0][0]

        # Generate precise_char_prob_score_map.
        precise_char_prob_feature = torch.sigmoid_(precise_char_prob_feature)
        precise_char_prob_score_map_mat = precise_char_prob_feature.numpy().astype(np.float32)

        # Force padding to be negative.
        if image.height < padded_image.height:
            pad_vert_begin = math.ceil(image.height / 2)
            if pad_vert_begin < precise_char_prob_score_map_mat.shape[0]:
                precise_char_prob_score_map_mat[pad_vert_begin:] = 0.0

        if image.width < padded_image.width:
            pad_hori_begin = math.ceil(image.width / 2)
            if pad_hori_begin < precise_char_prob_score_map_mat.shape[1]:
                precise_char_prob_score_map_mat[:, pad_hori_begin:] = 0.0

        precise_char_prob_score_map = ScoreMap(mat=precise_char_prob_score_map_mat)

        # Generate other score maps.
        precise_char_up_left_corner_offset_score_map = ScoreMap(
            mat=precise_char_up_left_corner_offset_feature.numpy().astype(np.float32),
            is_prob=False,
        )
        precise_char_corner_angle_score_map = ScoreMap(
            mat=precise_char_corner_angle_feature.numpy().astype(np.float32),
            is_prob=False,
        )
        precise_char_corner_distance_score_map = ScoreMap(
            mat=precise_char_corner_distance_feature.numpy().astype(np.float32),
            is_prob=False,
        )

        return AdaptiveScalingInferencingPresiceInferResult(
            padded_image=padded_image,
            precise_char_prob_score_map=precise_char_prob_score_map,
            precise_char_up_left_corner_offset_score_map=(
                precise_char_up_left_corner_offset_score_map
            ),
            precise_char_corner_angle_score_map=precise_char_corner_angle_score_map,
            precise_char_corner_distance_score_map=precise_char_corner_distance_score_map,
        )
