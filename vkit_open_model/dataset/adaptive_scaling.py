from typing import Tuple, Optional, Iterable, Dict, List, Any, Sequence, Mapping
import logging

import numpy as np
from numpy.random import Generator as RandomGenerator, default_rng
from torch.utils.data import IterableDataset, default_collate
import attrs

from vkit.element import Image, Mask, ScoreMap, Box
from vkit.utility import PathType, rng_shuffle, rng_choice_with_size
from vkit.pipeline import (
    pipeline_step_collection_factory,
    PageCroppingStepOutput,
    PageCharRegressionLabel,
    PageTextRegionCroppingStepOutput,
    PipelinePostProcessor,
    PipelinePostProcessorFactory,
    PipelineRunRngStateOutput,
    Pipeline,
    PipelinePool,
)

logger = logging.getLogger(__name__)


@attrs.define
class RoughSample:
    image: Image
    downsampled_shape: Tuple[int, int]
    downsampled_core_box: Box
    downsampled_mask: Mask
    downsampled_score_map: ScoreMap
    rng_state: Mapping


@attrs.define
class PreciseSample:
    image: Image
    downsampled_shape: Tuple[int, int]
    downsampled_core_box: Box
    downsampled_score_map: ScoreMap
    downsampled_page_char_regression_labels: Sequence[PageCharRegressionLabel]
    rng_state: Mapping


@attrs.define
class AdaptiveScalingPipelinePostProcessorConfig:
    precise_to_rough_ratio: float = 0.25


@attrs.define
class AdaptiveScalingPipelinePostProcessorInput:
    pipeline_run_rng_state_output: PipelineRunRngStateOutput
    page_cropping_step_output: PageCroppingStepOutput
    page_text_region_cropping_step_output: PageTextRegionCroppingStepOutput


class AdaptiveScalingPipelinePostProcessor(
    PipelinePostProcessor[  # type: ignore
        AdaptiveScalingPipelinePostProcessorConfig,
        AdaptiveScalingPipelinePostProcessorInput,
        Tuple[Sequence[RoughSample], Sequence[PreciseSample]],
    ]
):  # yapf: disable

    def generate_output(
        self,
        input: AdaptiveScalingPipelinePostProcessorInput,
        rng: RandomGenerator,
    ):
        rng_state = input.pipeline_run_rng_state_output.rng_state
        page_cropping_step_output = input.page_cropping_step_output
        page_text_region_cropping_step_output = input.page_text_region_cropping_step_output

        rough_samples: List[RoughSample] = []
        precise_samples: List[PreciseSample] = []

        for cropped_page in page_cropping_step_output.cropped_pages:
            downsampled_label = cropped_page.downsampled_label
            assert downsampled_label
            rough_samples.append(
                RoughSample(
                    image=cropped_page.page_image,
                    downsampled_shape=downsampled_label.shape,
                    downsampled_core_box=downsampled_label.core_box,
                    downsampled_mask=downsampled_label.page_char_mask,
                    downsampled_score_map=downsampled_label.page_char_height_score_map,
                    rng_state=rng_state,
                )
            )

        cropped_page_text_regions = rng_shuffle(
            rng,
            page_text_region_cropping_step_output.cropped_page_text_regions,
        )
        num_rough_to_precise = round(
            self.config.precise_to_rough_ratio * len(cropped_page_text_regions)
        )

        # Some will be transformed for rough prediction.
        for cropped_page_text_region in cropped_page_text_regions[:num_rough_to_precise]:
            downsampled_label = cropped_page_text_region.downsampled_label
            assert downsampled_label
            rough_samples.append(
                RoughSample(
                    image=cropped_page_text_region.page_image,
                    downsampled_shape=downsampled_label.shape,
                    downsampled_core_box=downsampled_label.core_box,
                    downsampled_mask=downsampled_label.page_char_mask,
                    downsampled_score_map=downsampled_label.page_char_height_score_map,
                    rng_state=rng_state,
                )
            )

        # And the rest will be used to train precise prediction.
        for cropped_page_text_region in cropped_page_text_regions[num_rough_to_precise:]:
            downsampled_label = cropped_page_text_region.downsampled_label
            assert downsampled_label
            precise_samples.append(
                PreciseSample(
                    image=cropped_page_text_region.page_image,
                    downsampled_shape=downsampled_label.shape,
                    downsampled_core_box=downsampled_label.core_box,
                    downsampled_score_map=downsampled_label.page_char_gaussian_score_map,
                    downsampled_page_char_regression_labels=(
                        downsampled_label.page_char_regression_labels
                    ),
                    rng_state=rng_state,
                )
            )

        return rough_samples, precise_samples


adaptive_scaling_pipeline_post_processor_factory = PipelinePostProcessorFactory(
    AdaptiveScalingPipelinePostProcessor
)


@attrs.define
class AdaptiveScalingIterableDatasetConfig:
    steps_json: PathType
    num_samples: int
    num_page_char_regression_labels: int
    rng_seed: int
    num_processes: int
    num_samples_reset_rng: Optional[int] = None
    num_cached_runs: Optional[int] = None
    is_dev: bool = False


class AdaptiveScalingIterableDataset(IterableDataset):

    def __init__(self, config: AdaptiveScalingIterableDatasetConfig):
        super().__init__()

        self.config = config

        logger.info('Creating pipeline pool...')
        num_runs_reset_rng = None
        if config.num_samples_reset_rng:
            num_runs_reset_rng = config.num_samples_reset_rng // config.num_processes

        self.pipeline_pool = PipelinePool(
            pipeline=Pipeline(
                steps=pipeline_step_collection_factory.create(config.steps_json),
                post_processor=adaptive_scaling_pipeline_post_processor_factory.create(),
            ),
            inventory=config.num_processes * 12,
            rng_seed=config.rng_seed,
            num_processes=config.num_processes,
            num_runs_reset_rng=num_runs_reset_rng,
        )
        logger.info('Pipeline pool created.')

        self.rng = default_rng(config.rng_seed)

        self.dev_rough_samples: List[RoughSample] = []
        self.dev_precise_samples: List[PreciseSample] = []

        if config.is_dev:
            while len(self.dev_rough_samples) < config.num_samples \
                    or len(self.dev_precise_samples) < config.num_samples:
                rough_samples, precise_samples = self.pipeline_pool.run()
                self.dev_rough_samples.extend(rough_samples)
                self.dev_precise_samples.extend(precise_samples)

            self.dev_rough_samples = self.dev_rough_samples[:config.num_samples]
            self.dev_precise_samples = self.dev_precise_samples[:config.num_samples]
            self.pipeline_pool.cleanup()

    def __iter__(self):
        if self.config.is_dev:
            assert len(self.dev_rough_samples) == self.config.num_samples
            yield from self.dev_rough_samples
            return

        cached_rough_samples: List[RoughSample] = []
        cached_precise_samples: List[PreciseSample] = []

        for _ in range(self.config.num_samples):

            if not cached_rough_samples or not cached_precise_samples:
                # Clear cache if either one is empty.
                cached_rough_samples.clear()
                cached_precise_samples.clear()

            while not cached_rough_samples or not cached_precise_samples:
                if not self.config.num_cached_runs:
                    rough_samples, precise_samples = self.pipeline_pool.run()
                    cached_rough_samples.extend(rough_samples)
                    cached_precise_samples.extend(precise_samples)

                else:
                    for _ in range(self.config.num_cached_runs):
                        rough_samples, precise_samples = self.pipeline_pool.run()
                        cached_rough_samples.extend(rough_samples)
                        cached_precise_samples.extend(precise_samples)

                    cached_rough_samples = list(rng_shuffle(self.rng, cached_rough_samples))
                    cached_precise_samples = list(rng_shuffle(self.rng, cached_precise_samples))

                if not cached_rough_samples or not cached_precise_samples:
                    logger.warning('cached_samples not filled!')

            rough_sample = cached_rough_samples.pop()
            precise_sample = cached_precise_samples.pop()

            # Sample char-level regression labels.
            downsampled_page_char_regression_labels = \
                precise_sample.downsampled_page_char_regression_labels
            precise_sample.downsampled_page_char_regression_labels = rng_choice_with_size(
                self.rng,
                downsampled_page_char_regression_labels,
                size=self.config.num_page_char_regression_labels,
                replace=(
                    len(downsampled_page_char_regression_labels) <  # noqa
                    self.config.num_page_char_regression_labels
                ),
            )

            yield (rough_sample, precise_sample)


def adaptive_scaling_dataset_collate_fn(batch: Iterable[Tuple[RoughSample, PreciseSample]]):
    default_rough_batch: List[Dict[str, np.ndarray]] = []
    rough_batch_downsampled_shape = None
    rough_batch_downsampled_core_box = None
    rough_batch_rng_states: List[Mapping] = []

    default_precise_batch: List[Dict[str, np.ndarray]] = []
    precise_batch_downsampled_shape = None
    precise_batch_downsampled_core_box = None
    precise_batch_rng_states: List[Mapping] = []

    for rough_sample, precise_sample in batch:
        default_rough_batch.append({
            # (H, W, 3) -> (3, H, W).
            'image': np.transpose(rough_sample.image.mat, axes=(2, 0, 1)).astype(np.float32),
            'downsampled_mask': rough_sample.downsampled_mask.np_mask.astype(np.float32),
            'downsampled_score_map': rough_sample.downsampled_score_map.mat,
        })
        rough_batch_downsampled_shape = rough_sample.downsampled_shape
        rough_batch_downsampled_core_box = rough_sample.downsampled_core_box
        rough_batch_rng_states.append(rough_sample.rng_state)

        downsampled_page_char_regression_labels = \
            precise_sample.downsampled_page_char_regression_labels
        # Label point, (P,)
        downsampled_label_point_y = np.array(
            [dpcrl.downsampled_label_point_y for dpcrl in downsampled_page_char_regression_labels],
            dtype=np.int64,
        )
        downsampled_label_point_x = np.array(
            [dpcrl.downsampled_label_point_x for dpcrl in downsampled_page_char_regression_labels],
            dtype=np.int64,
        )
        # Up-left corner, (P, 2), 2 for (y, x) offsets.
        up_left_offsets = np.array(
            [dpcrl.generate_up_left_offsets() for dpcrl in downsampled_page_char_regression_labels],
            dtype=np.int64,
        )
        # Corner angles, (P, 4)
        corner_angles = np.array(
            [
                dpcrl.generate_clockwise_angle_distribution()
                for dpcrl in downsampled_page_char_regression_labels
            ],
            dtype=np.float32,
        )
        # Corner distances, (P, 3)
        corner_distances = np.array(
            [
                dpcrl.generate_non_up_left_distances()
                for dpcrl in downsampled_page_char_regression_labels
            ],
            dtype=np.float32,
        )
        default_precise_batch.append({
            # (H, W, 3) -> (3, H, W).
            'image': np.transpose(precise_sample.image.mat, axes=(2, 0, 1)).astype(np.float32),
            'downsampled_score_map': precise_sample.downsampled_score_map.mat,
            'downsampled_label_point_y': downsampled_label_point_y,
            'downsampled_label_point_x': downsampled_label_point_x,
            'up_left_offsets': up_left_offsets,
            'corner_angles': corner_angles,
            'corner_distances': corner_distances,
        })
        precise_batch_downsampled_shape = precise_sample.downsampled_shape
        precise_batch_downsampled_core_box = precise_sample.downsampled_core_box
        precise_batch_rng_states.append(precise_sample.rng_state)

    assert rough_batch_downsampled_shape and rough_batch_downsampled_core_box
    rough_collated_batch: Dict[str, Any] = default_collate(default_rough_batch)
    rough_collated_batch['downsampled_shape'] = rough_batch_downsampled_shape
    rough_collated_batch['downsampled_core_box'] = rough_batch_downsampled_core_box
    rough_collated_batch['rng_states'] = rough_batch_rng_states

    assert precise_batch_downsampled_shape and precise_batch_downsampled_core_box
    precise_collated_batch: Dict[str, Any] = default_collate(default_precise_batch)
    precise_collated_batch['downsampled_shape'] = precise_batch_downsampled_shape
    precise_collated_batch['downsampled_core_box'] = precise_batch_downsampled_core_box
    precise_collated_batch['rng_states'] = precise_batch_rng_states

    return {
        'rough': rough_collated_batch,
        'precise': precise_collated_batch,
    }
