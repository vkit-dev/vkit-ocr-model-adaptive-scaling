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
from typing import Tuple, List, Sequence
import math
import logging

import torch
import numpy as np
from scipy.ndimage import maximum_filter
import cv2 as cv
import iolite as io
import attrs

from vkit.utility import PathType
from vkit.element import Point, PointTuple, Box, Polygon, Mask, ScoreMap, Image
from vkit.mechanism.distortion.geometric.affine import (
    affine_polygons,
    RotateConfig,
    RotateState,
)
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
    precise_flattened_text_region_resized_char_height_median: int = 32
    precise_flattened_text_region_resized_ratio_min: float = 0.25
    precise_stack_flattened_text_regions_page_pad: int = 10
    precise_stack_flattened_text_regions_pad: int = 2
    precise_char_mask_positive_thr: float = 0.5
    precise_build_polygons_positive_char_prob_thr: float = 0.8
    precise_build_polygons_maximum_filter_size: float = 10


@attrs.define
class AdaptiveScalingInferencingRoughInferResult:
    resized_shape: Tuple[int, int]
    padded_image: Image
    rough_char_mask: Mask
    rough_char_height_score_map: ScoreMap


@attrs.define
class AdaptiveScalingInferencingPresiceInferResult:
    padded_image: Image
    precise_char_mask: Mask
    precise_char_prob_score_map: ScoreMap
    precise_np_char_up_left_corner_offset: np.ndarray
    precise_np_char_corner_angle_distribution: np.ndarray
    precise_np_char_corner_distance: np.ndarray


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
                rough_char_height_feature,
            ) = self.model_jit.forward_rough(x)  # type: ignore

        # (1, 1, H / 2, D / 2)) -> (H / 2, D / 2))
        rough_char_mask_feature = rough_char_mask_feature[0][0]
        rough_char_height_feature = rough_char_height_feature[0][0]
        # Assert shape.
        assert rough_char_mask_feature.shape == rough_char_height_feature.shape
        assert rough_char_mask_feature.shape[0] == padded_image.height // 2
        assert rough_char_mask_feature.shape[1] == padded_image.width // 2

        # Generate rough_char_mask & rough_char_height_score_map.
        rough_char_mask_feature = torch.sigmoid_(rough_char_mask_feature)
        rough_char_mask_feature = torch.greater_equal(
            rough_char_mask_feature,
            self.config.rough_char_mask_positive_thr,
        )
        rough_char_mask_mat = rough_char_mask_feature.numpy().astype(np.uint8)

        rough_char_height_score_map_mat = rough_char_height_feature.numpy().astype(np.float32)

        # Force padding to be negative.
        if image.height < padded_image.height:
            pad_vert_begin = math.ceil(image.height / 2)
            if pad_vert_begin < rough_char_mask_mat.shape[0]:
                rough_char_mask_mat[pad_vert_begin:] = 0
                rough_char_height_score_map_mat[pad_vert_begin:] = 0.0

        if image.width < padded_image.width:
            pad_hori_begin = math.ceil(image.width / 2)
            if pad_hori_begin < rough_char_mask_mat.shape[1]:
                rough_char_mask_mat[:, pad_hori_begin:] = 0
                rough_char_height_score_map_mat[:, pad_hori_begin:] = 0.0

        # Clear char height that is too small.
        np_invalid_mask = rough_char_height_score_map_mat < self.config.rough_valid_char_height_min
        rough_char_height_score_map_mat[np_invalid_mask] = 0.0

        rough_char_mask = Mask(mat=rough_char_mask_mat)
        rough_char_height_score_map = ScoreMap(
            mat=rough_char_height_score_map_mat,
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
            rough_char_height_score_map=rough_char_height_score_map,
        )

    def build_flattened_text_regions(
        self,
        image: Image,
        rough_infer_result: AdaptiveScalingInferencingRoughInferResult,
    ):
        resized_shape = rough_infer_result.resized_shape
        resized_height, _ = resized_shape
        rough_char_mask = rough_infer_result.rough_char_mask
        rough_char_height_score_map = rough_infer_result.rough_char_height_score_map

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
            char_height_score_map = rough_polygon.extract_score_map(rough_char_height_score_map)
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
                precise_char_mask_feature,
                precise_char_prob_feature,
                precise_char_up_left_corner_offset_feature,
                precise_char_corner_angle_feature,
                precise_char_corner_distance_feature,
            ) = self.model_jit.forward_precise(x)  # type: ignore

        # (1, 1, H / 2, D / 2)) -> (H / 2, D / 2))
        precise_char_mask_feature = precise_char_mask_feature[0][0]
        precise_char_prob_feature = precise_char_prob_feature[0][0]

        # (1, 2, H / 2, D / 2)) -> (H / 2, D / 2, 2))
        precise_char_up_left_corner_offset_feature = \
            torch.permute(precise_char_up_left_corner_offset_feature[0], [1, 2, 0])
        assert precise_char_up_left_corner_offset_feature.shape[-1] == 2

        # (1, 4, H / 2, D / 2)) -> (H / 2, D / 2, 4))
        precise_char_corner_angle_feature = \
            torch.permute(precise_char_corner_angle_feature[0], [1, 2, 0])
        assert precise_char_corner_angle_feature.shape[-1] == 4

        # (1, 3, H / 2, D / 2)) -> (H / 2, D / 2, 3))
        precise_char_corner_distance_feature = \
            torch.permute(precise_char_corner_distance_feature[0], [1, 2, 0])
        assert precise_char_corner_distance_feature.shape[-1] == 3

        # Generate precise_char_mask.
        precise_char_mask_feature = torch.sigmoid_(precise_char_mask_feature)
        precise_char_mask_feature = torch.greater_equal(
            precise_char_mask_feature,
            self.config.precise_char_mask_positive_thr,
        )
        precise_char_mask_mat = precise_char_mask_feature.numpy().astype(np.uint8)

        # Generate precise_char_prob_score_map.
        precise_char_prob_feature = torch.sigmoid_(precise_char_prob_feature)
        precise_char_prob_score_map_mat = precise_char_prob_feature.numpy().astype(np.float32)

        # Force padding to be negative.
        if image.height < padded_image.height:
            pad_vert_begin = math.ceil(image.height / 2)
            if pad_vert_begin < precise_char_prob_score_map_mat.shape[0]:
                precise_char_mask_mat[pad_vert_begin:] = 0
                precise_char_prob_score_map_mat[pad_vert_begin:] = 0.0

        if image.width < padded_image.width:
            pad_hori_begin = math.ceil(image.width / 2)
            if pad_hori_begin < precise_char_prob_score_map_mat.shape[1]:
                precise_char_mask_mat[:, pad_hori_begin:] = 0
                precise_char_prob_score_map_mat[:, pad_hori_begin:] = 0.0

        precise_char_mask = Mask(mat=precise_char_mask_mat)
        precise_char_prob_score_map = ScoreMap(mat=precise_char_prob_score_map_mat)

        # Generate other np arrays.
        precise_np_char_up_left_corner_offset = \
            precise_char_up_left_corner_offset_feature.numpy().astype(np.float32)

        precise_char_corner_angle_feature = \
            torch.softmax(precise_char_corner_angle_feature, axis=-1)  # type: ignore
        precise_np_char_corner_angle_distribution = \
            precise_char_corner_angle_feature.numpy().astype(np.float32)

        precise_np_char_corner_distance = \
            precise_char_corner_distance_feature.numpy().astype(np.float32)

        return AdaptiveScalingInferencingPresiceInferResult(
            padded_image=padded_image,
            precise_char_mask=precise_char_mask,
            precise_char_prob_score_map=precise_char_prob_score_map,
            precise_np_char_up_left_corner_offset=precise_np_char_up_left_corner_offset,
            precise_np_char_corner_angle_distribution=precise_np_char_corner_angle_distribution,
            precise_np_char_corner_distance=precise_np_char_corner_distance,
        )

    @classmethod
    def precise_build_polygon(
        cls,
        precise_infer_result: AdaptiveScalingInferencingPresiceInferResult,
        point: Point,
    ):
        padded_image = precise_infer_result.padded_image
        precise_np_char_up_left_corner_offset = \
            precise_infer_result.precise_np_char_up_left_corner_offset
        precise_np_char_corner_angle_distribution = \
            precise_infer_result.precise_np_char_corner_angle_distribution
        precise_np_char_corner_distance = \
            precise_infer_result.precise_np_char_corner_distance

        # NOTE: point is in downsampled space.
        upsampled_point = point.to_conducted_resized_point(
            precise_np_char_up_left_corner_offset.shape[:2],
            resized_height=padded_image.height,
            resized_width=padded_image.width,
        )

        # Locate up-left corner.
        (
            up_left_offset_y,
            up_left_offset_x,
        ) = precise_np_char_up_left_corner_offset[point.y][point.x]

        up_left = Point.create(
            y=upsampled_point.smooth_y + up_left_offset_y,
            x=upsampled_point.smooth_x + up_left_offset_x,
        )

        # Get angles.
        angle_distrib = precise_np_char_corner_angle_distribution[point.y][point.x]
        assert len(angle_distrib) == 4

        # Get other corner distances.
        other_corner_distances = precise_np_char_corner_distance[point.y][point.x]
        assert len(other_corner_distances) == 3
        up_right_dis, down_right_dis, down_left_dis = other_corner_distances

        # Build other corner points.
        theta = np.arctan2(up_left_offset_y, up_left_offset_x)
        two_pi = 2 * np.pi
        theta = theta % two_pi

        theta += angle_distrib[0] * two_pi
        theta = theta % two_pi
        up_right = Point.create(
            y=upsampled_point.smooth_y + np.sin(theta) * up_right_dis,
            x=upsampled_point.smooth_x + np.cos(theta) * up_right_dis,
        )

        theta += angle_distrib[1] * two_pi
        theta = theta % two_pi
        down_right = Point.create(
            y=upsampled_point.smooth_y + np.sin(theta) * down_right_dis,
            x=upsampled_point.smooth_x + np.cos(theta) * down_right_dis,
        )

        theta += angle_distrib[2] * two_pi
        theta = theta % two_pi
        down_left = Point.create(
            y=upsampled_point.smooth_y + np.sin(theta) * down_left_dis,
            x=upsampled_point.smooth_x + np.cos(theta) * down_left_dis,
        )

        return Polygon.create([up_left, up_right, down_right, down_left])

    def precise_build_grouped_polygons(
        self,
        precise_infer_result: AdaptiveScalingInferencingPresiceInferResult,
        flattened_text_regions: Sequence[FlattenedTextRegion],
        boxes: Sequence[Box],
    ):
        padded_image = precise_infer_result.padded_image
        precise_char_mask = precise_infer_result.precise_char_mask
        precise_char_prob_score_map = precise_infer_result.precise_char_prob_score_map

        assert len(flattened_text_regions) == len(boxes)

        # Find the peaks.
        mat = precise_char_prob_score_map.mat.copy()

        mat[~precise_char_mask.np_mask] = 0

        np_local_maximum = maximum_filter(
            mat,
            size=self.config.precise_build_polygons_maximum_filter_size,
        )
        np_mask = (np_local_maximum == mat)

        np_mask[mat < self.config.precise_build_polygons_positive_char_prob_thr] = 0

        # Convert peaks to points.
        grouped_points: List[PointTuple] = []
        for flattened_text_region, box in zip(flattened_text_regions, boxes):
            assert flattened_text_region.shape == box.shape
            downsampled_box = box.to_conducted_resized_box(
                padded_image,
                resized_height=precise_char_prob_score_map.height,
                resized_width=precise_char_prob_score_map.width,
            )
            downsampled_flattened_mask = flattened_text_region.flattened_mask.to_resized_mask(
                resized_height=downsampled_box.height,
                resized_width=downsampled_box.width,
            )

            np_boxed_mask = downsampled_box.extract_np_array(np_mask)
            np_boxed_mask[~downsampled_flattened_mask.np_mask] = 0

            np_boxed_ys, np_boxed_xs = np.nonzero(np_boxed_mask)
            boxed_points = PointTuple.from_np_array(np.column_stack((np_boxed_xs, np_boxed_ys)))
            points = boxed_points.to_shifted_points(
                offset_y=downsampled_box.up,
                offset_x=downsampled_box.left,
            )
            grouped_points.append(points)

        # Build polygons.
        grouped_polygons: List[List[Polygon]] = []
        for points in grouped_points:
            polygons = [self.precise_build_polygon(precise_infer_result, point) for point in points]
            grouped_polygons.append(polygons)

        return grouped_polygons

    @classmethod
    def precise_build_remapped_polygons(
        cls,
        flattened_text_regions: Sequence[FlattenedTextRegion],
        boxes: Sequence[Box],
        grouped_polygons: Sequence[Sequence[Polygon]],
    ):
        remapped_polygons: List[Polygon] = []

        np_trans_mat_last_row = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)

        assert len(flattened_text_regions) == len(boxes) == len(grouped_polygons)
        for flattened_text_region, box, polygons in zip(
            flattened_text_regions, boxes, grouped_polygons
        ):
            if not polygons:
                continue

            assert flattened_text_region.shape == box.shape

            # 1. Undo resize and trim.
            height_before_resize, width_before_resize = flattened_text_region.shape_before_resize
            rotated_trimmed_box = flattened_text_region.rotated_trimmed_box
            assert flattened_text_region.post_rotate_angle == 0

            after_rotate_remapped_polygons: List[Polygon] = []
            for polygon in polygons:
                polygon = polygon.to_relative_polygon(
                    origin_y=box.up,
                    origin_x=box.left,
                )
                polygon = polygon.to_conducted_resized_polygon(
                    flattened_text_region.shape,
                    resized_height=height_before_resize,
                    resized_width=width_before_resize,
                )
                polygon = polygon.to_shifted_polygon(
                    offset_y=rotated_trimmed_box.up,
                    offset_x=rotated_trimmed_box.left,
                )
                after_rotate_remapped_polygons.append(polygon)

            # 2. Undo rotate.
            bounding_extended_text_region_box = \
                flattened_text_region.bounding_extended_text_region_mask.box
            assert bounding_extended_text_region_box

            flattening_rotate_angle = flattened_text_region.flattening_rotate_angle

            rotate_state = RotateState(
                config=RotateConfig(flattening_rotate_angle),
                shape=bounding_extended_text_region_box.shape,
                rng=None,
            )
            trans_mat = rotate_state.trans_mat
            assert trans_mat.shape == (2, 3)

            trans_mat = np.vstack((trans_mat, np_trans_mat_last_row))
            inv_trans_mat = np.linalg.inv(trans_mat)

            before_rotate_remapped_polygons = affine_polygons(
                inv_trans_mat,
                after_rotate_remapped_polygons,
            )

            # 3. Shift back.
            for polygon in before_rotate_remapped_polygons:
                remapped_polygons.append(
                    polygon.to_shifted_polygon(
                        offset_y=bounding_extended_text_region_box.up,
                        offset_x=bounding_extended_text_region_box.left,
                    )
                )

        return remapped_polygons
