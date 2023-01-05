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
from typing import Sequence, Mapping

import torch
from torch.utils.data import DataLoader
import numpy as np
import iolite as io

from vkit.element import PointList, Point, Box, Mask, ScoreMap, Image
from vkit.mechanism.painter import Painter
from vkit_open_model.model.adaptive_scaling import (
    AdaptiveScalingSize,
    AdaptiveScalingNeckHeadType,
    AdaptiveScalingConfig,
    AdaptiveScaling,
)
from vkit_open_model.dataset.adaptive_scaling import (
    adaptive_scaling_dataset_collate_fn,
    AdaptiveScalingIterableDatasetConfig,
    AdaptiveScalingIterableDataset,
)
from vkit_open_model.loss_function import (
    AdaptiveScalingRoughLossFunctionConifg,
    AdaptiveScalingRoughLossFunction,
    AdaptiveScalingPreciseLossFunctionConifg,
    AdaptiveScalingPreciseLossFunction,
)


def debug_adaptive_scaling_jit():
    model = AdaptiveScaling(
        AdaptiveScalingConfig(AdaptiveScalingSize.TINY, AdaptiveScalingNeckHeadType.UPERNEXT)
    )
    model_jit = torch.jit.script(model)  # type: ignore

    x = torch.rand((1, 3, 320, 320))
    (
        rough_char_mask_feature,
        rough_char_height_feature,
    ) = model_jit.forward_rough(x)  # type: ignore
    assert rough_char_mask_feature.shape == (1, 1, 160, 160)
    assert rough_char_height_feature.shape == (1, 1, 160, 160)

    (
        precise_char_prob_feature,
        precise_char_up_left_corner_feature,
        precise_char_corner_angle_feature,
        precise_char_corner_distance_feature,
    ) = model_jit.forward_precise(x)  # type: ignore
    assert precise_char_prob_feature.shape == (1, 1, 160, 160)
    assert precise_char_up_left_corner_feature.shape == (1, 2, 160, 160)
    assert precise_char_corner_angle_feature.shape == (1, 4, 160, 160)
    assert precise_char_corner_distance_feature.shape == (1, 3, 160, 160)


# def debug_adaptive_scaling_jit_model_summary():
#     from torchinfo import summary
#     print('AdaptiveScaling(BASE, FPN)')
#     model = AdaptiveScaling(
#         AdaptiveScalingConfig(AdaptiveScalingSize.BASE, AdaptiveScalingNeckHeadType.FPN)
#     )
#     model.eval()
#     print('depth=1')
#     summary(model, input_size=(1, 3, 640, 40), depth=1)
#     print('depth=3')
#     summary(model, input_size=(1, 3, 640, 40), depth=3)
#     print()

#     print('AdaptiveScaling(BASE, UPERNEXT)')
#     model = AdaptiveScaling(
#         AdaptiveScalingConfig(AdaptiveScalingSize.BASE, AdaptiveScalingNeckHeadType.UPERNEXT)
#     )
#     model.eval()
#     print('depth=1')
#     summary(model, input_size=(1, 3, 640, 40), depth=1)
#     print('depth=3')
#     summary(model, input_size=(1, 3, 640, 40), depth=3)
#     print()


def test_get_label_point_feature():
    feature = torch.rand((2, 4, 640, 320))
    label_point_y = torch.randint(low=0, high=640, size=(2, 20))
    label_point_x = torch.randint(low=0, high=320, size=(2, 20))
    label_point_feature = AdaptiveScalingPreciseLossFunction.get_label_point_feature(
        feature=feature,
        label_point_y=label_point_y,
        label_point_x=label_point_x,
    )
    assert label_point_feature.shape == (2, 20, 4)


def profile_adaptive_scaling_jit_forward():
    model = AdaptiveScaling(AdaptiveScalingConfig(AdaptiveScalingSize.TINY))
    model_jit = torch.jit.script(model)  # type: ignore
    del model

    with torch.autograd.profiler.profile() as prof:
        x = torch.rand((2, 3, 640, 640))
        model_jit.forward_rough(x)  # type: ignore
    print('ROUGH:', prof.key_averages().table(sort_by="self_cpu_time_total"))  # type: ignore

    with torch.autograd.profiler.profile() as prof:
        x = torch.rand((2, 3, 640, 640))
        model_jit.forward_precise(x)  # type: ignore
    print('PRECISE:', prof.key_averages().table(sort_by="self_cpu_time_total"))  # type: ignore


def debug_adaptive_scaling_jit_loss_backward():
    model = AdaptiveScaling(AdaptiveScalingConfig(AdaptiveScalingSize.TINY))
    model_jit = torch.jit.script(model)  # type: ignore
    del model

    loss_function = AdaptiveScalingRoughLossFunction(AdaptiveScalingRoughLossFunctionConifg())

    x = torch.randint(low=0, high=256, size=(2, 3, 640, 640)).to(torch.float32)
    (
        rough_char_mask_feature,
        rough_char_height_feature,
    ) = model_jit.forward_rough(x)  # type: ignore

    loss = loss_function(
        rough_char_mask_feature=rough_char_mask_feature,
        rough_char_height_feature=rough_char_height_feature,
        downsampled_mask=(torch.rand(2, 300, 300) > 0.5).float(),
        downsampled_score_map=torch.rand(2, 300, 300) + 8.75,
        downsampled_shape=(320, 320),
        downsampled_core_box=Box(up=10, down=309, left=10, right=309),
    )
    loss.backward()

    rough_name_to_grad = AdaptiveScaling.debug_get_rough_name_to_grad(model_jit)  # type: ignore

    loss_function = AdaptiveScalingPreciseLossFunction(AdaptiveScalingPreciseLossFunctionConifg())

    x = torch.randint(low=0, high=256, size=(2, 3, 640, 640)).to(torch.float32)
    (
        precise_char_mask_feature,
        precise_char_prob_feature,
        precise_char_up_left_corner_offset_feature,
        precise_char_corner_angle_feature,
        precise_char_corner_distance_feature,
    ) = model_jit.forward_precise(x)  # type: ignore

    loss = loss_function(
        precise_char_mask_feature=precise_char_mask_feature,
        precise_char_prob_feature=precise_char_prob_feature,
        precise_char_up_left_corner_offset_feature=precise_char_up_left_corner_offset_feature,
        precise_char_corner_angle_feature=precise_char_corner_angle_feature,
        precise_char_corner_distance_feature=precise_char_corner_distance_feature,
        downsampled_char_prob_score_map=torch.rand(2, 300, 300),
        downsampled_shape=(320, 320),
        downsampled_core_box=Box(up=10, down=309, left=10, right=309),
        downsampled_char_mask=(torch.rand(2, 300, 300) > 0.5).float(),
        downsampled_label_point_y=torch.randint(low=10, high=310, size=(2, 20)),
        downsampled_label_point_x=torch.randint(low=10, high=310, size=(2, 20)),
        char_up_left_offsets=torch.randint(low=-20, high=21, size=(2, 20, 2)),
        char_corner_angles=torch.rand((2, 20, 4)),
        char_corner_distances=torch.rand((2, 20, 3)),
    )
    loss.backward()

    precise_name_to_grad = \
        AdaptiveScaling.debug_get_precise_name_to_grad(model_jit, rough_name_to_grad)  # type: ignore # noqa

    AdaptiveScaling.debug_inspect_name_to_grad(rough_name_to_grad, precise_name_to_grad)


def sample_adaptive_scaling_dataset(
    num_processes: int,
    num_page_char_regression_labels: int,
    batch_size: int,
    epoch_size: int,
    output_folder: str,
):
    out_fd = io.folder(output_folder, touch=True)

    num_samples = batch_size * epoch_size
    dataset = AdaptiveScalingIterableDataset(
        AdaptiveScalingIterableDatasetConfig(
            steps_json=(
                '$VKIT_ARTIFACT_PACK/pipeline/text_detection/dev_adaptive_scaling_dataset_steps.json'  # noqa
            ),
            num_samples=num_samples,
            num_page_char_regression_labels=num_page_char_regression_labels,
            rng_seed=13,
            num_processes=num_processes,
        )
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=adaptive_scaling_dataset_collate_fn,
    )

    for batch_idx, batch in enumerate(data_loader):
        print(f'Saving batch_idx={batch_idx} ...')

        # Rough.
        rough_batch = batch['rough']
        batch_image = rough_batch['image']
        batch_downsampled_mask = rough_batch['downsampled_mask']
        batch_downsampled_score_map = rough_batch['downsampled_score_map']
        downsampled_shape = rough_batch['downsampled_shape']
        downsampled_core_box: Box = rough_batch['downsampled_core_box']
        rng_states: Sequence[Mapping] = rough_batch['rng_states']

        assert batch_image.shape[0] \
            == batch_downsampled_mask.shape[0] \
            == batch_downsampled_score_map.shape[0] \
            == batch_size

        for sample_in_batch_idx in range(batch_size):
            np_image = \
                torch.permute(batch_image[sample_in_batch_idx], (1, 2, 0)).numpy().astype(np.uint8)
            np_downsampled_mask = \
                batch_downsampled_mask[sample_in_batch_idx].numpy().astype(np.uint8)
            np_downsampled_score_map = \
                batch_downsampled_score_map[sample_in_batch_idx].numpy()

            image = Image(mat=np_image)
            downsampled_mask = Mask(
                mat=np_downsampled_mask,
                box=downsampled_core_box,
            )
            downsampled_score_map = ScoreMap(
                mat=np_downsampled_score_map,
                is_prob=False,
                box=downsampled_core_box,
            )

            output_prefix = f'{batch_idx}_{sample_in_batch_idx}_rough'
            image.to_file(out_fd / f'{output_prefix}_image.jpg')

            downsampled_image = image.to_resized_image(
                resized_height=downsampled_shape[0],
                resized_width=downsampled_shape[1],
            )

            painter = Painter.create(downsampled_image)
            painter.paint_mask(downsampled_mask)
            painter.to_file(out_fd / f'{output_prefix}_downsampled_mask.jpg')

            painter = Painter.create(downsampled_image)
            painter.paint_score_map(downsampled_score_map)
            painter.to_file(out_fd / f'{output_prefix}_downsampled_score_map.jpg')

            painter = Painter.create(downsampled_image)
            painter.paint_score_map(downsampled_score_map, alpha=1.0)
            painter.to_file(out_fd / f'{output_prefix}_downsampled_score_map_alpha_1.jpg')

            rng_state = rng_states[sample_in_batch_idx]
            io.write_json(out_fd / f'{output_prefix}_rng_state.json', rng_state)

        # Precise.
        precise_batch = batch['precise']
        batch_image = precise_batch['image']
        batch_downsampled_score_map = precise_batch['downsampled_score_map']
        batch_downsampled_label_point_y = precise_batch['downsampled_label_point_y']
        batch_downsampled_label_point_x = precise_batch['downsampled_label_point_x']
        batch_up_left_offsets = precise_batch['up_left_offsets']
        downsampled_shape = precise_batch['downsampled_shape']
        downsampled_core_box: Box = precise_batch['downsampled_core_box']

        assert batch_image.shape[0] \
            == batch_downsampled_label_point_y.shape[0] \
            == batch_downsampled_label_point_x.shape[0] \
            == batch_size

        for sample_in_batch_idx in range(batch_size):
            np_image = \
                torch.permute(batch_image[sample_in_batch_idx], (1, 2, 0)).numpy().astype(np.uint8)
            image = Image(mat=np_image)

            np_downsampled_score_map = \
                batch_downsampled_score_map[sample_in_batch_idx].numpy()
            downsampled_score_map = ScoreMap(
                mat=np_downsampled_score_map,
                box=downsampled_core_box,
            )

            output_prefix = f'{batch_idx}_{sample_in_batch_idx}_precise'
            image.to_file(out_fd / f'{output_prefix}_image.jpg')

            downsampled_image = image.to_resized_image(
                resized_height=downsampled_shape[0],
                resized_width=downsampled_shape[1],
            )

            painter = Painter.create(downsampled_image)
            painter.paint_score_map(downsampled_score_map)
            painter.to_file(out_fd / f'{output_prefix}_downsampled_score_map.jpg')

            np_downsampled_label_point_y = \
                batch_downsampled_label_point_y[sample_in_batch_idx].numpy().astype(np.int32)
            np_downsampled_label_point_x = \
                batch_downsampled_label_point_x[sample_in_batch_idx].numpy().astype(np.int32)

            downsampled_label_points = PointList()
            for y, x in zip(np_downsampled_label_point_y, np_downsampled_label_point_x):
                downsampled_label_points.append(Point.create(y=y, x=x))

            downsampled_up_left_corner_points = PointList()
            np_up_left_offsets = batch_up_left_offsets[sample_in_batch_idx].numpy()
            for (y_offset, x_offset), y, x in zip(
                np_up_left_offsets,
                np_downsampled_label_point_y,
                np_downsampled_label_point_x,
            ):
                y += y_offset / 2
                x += x_offset / 2
                downsampled_up_left_corner_points.append(Point.create(y=y, x=x))

            painter = Painter.create(downsampled_image)
            painter.paint_points(downsampled_label_points, color='red', radius=2, alpha=0.9)
            painter.paint_points(
                downsampled_up_left_corner_points,
                color='blue',
                radius=2,
                alpha=0.9,
            )
            painter.to_file(out_fd / f'{output_prefix}_downsampled_label_points.jpg')


def profile_adaptive_scaling_dataset(
    num_processes: int,
    num_page_char_regression_labels: int,
    batch_size: int,
    epoch_size: int,
):
    from datetime import datetime
    from tqdm import tqdm
    import numpy as np

    num_samples = batch_size * epoch_size

    data_loader = DataLoader(
        dataset=AdaptiveScalingIterableDataset(
            AdaptiveScalingIterableDatasetConfig(
                steps_json='$VKIT_ARTIFACT_PACK/pipeline/text_detection/adaptive_scaling.json',
                num_samples=num_samples,
                num_page_char_regression_labels=num_page_char_regression_labels,
                rng_seed=13370,
                num_processes=num_processes,
            )
        ),
        batch_size=batch_size,
        collate_fn=adaptive_scaling_dataset_collate_fn,
    )

    dt_batches = []
    dt_begin = datetime.now()
    for _ in tqdm(data_loader):
        dt_batches.append(datetime.now())
    dt_end = datetime.now()

    dt_delta = dt_end - dt_begin
    print('total:', dt_delta.seconds)
    print('per_batch:', dt_delta.seconds / batch_size)
    dt_batch_deltas = []
    for idx, dt_batch in enumerate(dt_batches):
        if idx == 0:
            dt_prev = dt_begin
        else:
            dt_prev = dt_batches[idx - 1]
        dt_batch_deltas.append((dt_batch - dt_prev).seconds)
    print('per_batch std:', float(np.std(dt_batch_deltas)))
