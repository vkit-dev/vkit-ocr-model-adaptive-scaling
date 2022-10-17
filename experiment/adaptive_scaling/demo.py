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
from typing import Sequence
from pathlib import Path

import torch
import cv2 as cv
import iolite as io

from vkit.utility import dyn_structure
from vkit.element import Image, Mask, ScoreMap, Painter  # noqa
from vkit.pipeline.text_detection.page_text_region import FlattenedTextRegion
from vkit_open_model.inferencing.adaptive_scaling import (
    AdaptiveScalingInferencingConfig,
    AdaptiveScalingInferencingRoughInferResult,
    AdaptiveScalingInferencingPresiceInferResult,
    AdaptiveScalingInferencing,
)


def visualize_rough_infer_result(
    out_fd: Path,
    rough_infer_result: AdaptiveScalingInferencingRoughInferResult,
):
    padded_image = rough_infer_result.padded_image
    rough_char_mask = rough_infer_result.rough_char_mask
    rough_char_height_score_map = rough_infer_result.rough_char_height_score_map

    rough_char_mask = rough_char_mask.to_resized_mask(
        resized_height=padded_image.height,
        resized_width=padded_image.width,
        cv_resize_interpolation=cv.INTER_NEAREST,
    )
    rough_char_height_score_map = rough_char_height_score_map.to_resized_score_map(
        resized_height=padded_image.height,
        resized_width=padded_image.width,
        cv_resize_interpolation=cv.INTER_NEAREST,
    )

    painter = Painter(padded_image)
    painter.paint_mask(rough_char_mask)
    painter.to_file(out_fd / 'rough_mask.jpg')

    painter = Painter(padded_image)
    painter.paint_score_map(rough_char_height_score_map, alpha=1.0)
    render_image = padded_image.copy()
    render_image[rough_char_mask] = painter.image
    render_image.to_file(out_fd / 'rough_score_map.jpg')


def visualize_text_regions(
    out_fd: Path,
    image: Image,
    flattened_text_regions: Sequence[FlattenedTextRegion],
    stacked_image: Image,
):
    polygons = [
        flattened_text_region.text_region_polygon
        for flattened_text_region in flattened_text_regions
    ]
    painter = Painter(image)
    painter.paint_polygons(polygons)
    painter.to_file(out_fd / 'text_region_polygons.jpg')

    if False:
        sub_out_fd = io.folder(out_fd / 'text_regions', touch=True)
        for idx, text_region in enumerate(flattened_text_regions):
            text_region.flattened_image.to_file(sub_out_fd / f'{idx}.jpg')

    stacked_image.to_file(out_fd / 'stacked_image.jpg')


def visualize_precise_infer_result(
    out_fd: Path,
    precise_infer_result: AdaptiveScalingInferencingPresiceInferResult,
):
    padded_image = precise_infer_result.padded_image
    precise_char_prob_score_map = precise_infer_result.precise_char_prob_score_map

    precise_char_prob_score_map = precise_char_prob_score_map.to_resized_score_map(
        resized_height=padded_image.height,
        resized_width=padded_image.width,
        cv_resize_interpolation=cv.INTER_NEAREST,
    )

    painter = Painter(padded_image)
    painter.paint_score_map(precise_char_prob_score_map)
    painter.to_file(out_fd / 'precise_char_prob_score_map.jpg')

    painter = Painter(padded_image)
    painter.paint_mask(precise_char_prob_score_map.to_mask(0.7))
    painter.to_file(out_fd / 'precise_char_prob_gt_70_mask.jpg')


def infer(
    inferencing_config_json: str,
    image_file: str,
    output_folder: str,
):
    out_fd = io.folder(output_folder, touch=True)

    inferencing_config = dyn_structure(
        inferencing_config_json,
        AdaptiveScalingInferencingConfig,
        support_path_type=True,
    )
    inferencing = AdaptiveScalingInferencing(inferencing_config)

    image = Image.from_file(image_file)

    rough_infer_result = inferencing.rough_infer(image)
    visualize_rough_infer_result(out_fd, rough_infer_result)

    flattened_text_regions = inferencing.build_flattened_text_regions(image, rough_infer_result)
    stacked_image, stacked_boxes = inferencing.stack_flattened_text_regions(flattened_text_regions)
    assert stacked_boxes  # TODO
    visualize_text_regions(out_fd, image, flattened_text_regions, stacked_image)

    precise_infer_result = inferencing.precise_infer(stacked_image)
    visualize_precise_infer_result(out_fd, precise_infer_result)


def convert_model_jit_to_model_onnx(
    model_jit_path: str,
    model_onnx_path: str,
):
    model_jit: torch.jit.ScriptModule = torch.jit.load(model_jit_path)  # type: ignore
    model_jit = model_jit.eval()
    torch.set_grad_enabled(False)

    torch.onnx.export(
        model_jit,
        torch.randn(1, 3, 640, 640),
        model_onnx_path,
        input_names=['x'],
        dynamic_axes={
            'x': {
                0: 'batch_size',
                2: 'height',
                3: 'width',
            },
        }
    )
