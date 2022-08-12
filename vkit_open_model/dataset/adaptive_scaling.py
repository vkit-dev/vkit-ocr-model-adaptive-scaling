from typing import Tuple, Optional, Iterable, Dict, List, Any, Sequence
import logging

import numpy as np
from numpy.random import Generator as RandomGenerator, default_rng
from torch.utils.data import IterableDataset, default_collate
import attrs

from vkit.element import Image, Mask, ScoreMap, Box
from vkit.utility import PathType
from vkit.pipeline import (
    PipelineState,
    PageCroppingStep,
    NoneTypePipelinePostProcessorConfig,
    PipelinePostProcessor,
    PipelinePostProcessorFactory,
    Pipeline,
    PipelinePool,
    pipeline_step_collection_factory,
)

logger = logging.getLogger(__name__)

Sample = Tuple[Image, Tuple[int, int], Box, Mask, ScoreMap]


class AdaptiveScalingPipelinePostProcessor(
    PipelinePostProcessor[
        NoneTypePipelinePostProcessorConfig,
        Sequence[Sample],
    ]
):  # yapf: disable

    def generate_output(self, state: PipelineState, rng: RandomGenerator):
        page_cropping_step_output = state.get_pipeline_step_output(PageCroppingStep)
        samples: List[Sample] = []
        for cropped_page in page_cropping_step_output.cropped_pages:
            downsampled_label = cropped_page.downsampled_label
            assert downsampled_label
            samples.append((
                cropped_page.page_image,
                downsampled_label.shape,
                downsampled_label.core_box,
                downsampled_label.page_char_mask,
                downsampled_label.page_char_height_score_map,
            ))
        return samples


adaptive_scaling_pipeline_post_processor_factory = PipelinePostProcessorFactory(
    AdaptiveScalingPipelinePostProcessor
)


@attrs.define
class AdaptiveScalingIterableDatasetConfig:
    steps_json: PathType
    num_samples: int
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

        self.dev_samples: List[Sample] = []
        if config.is_dev:
            while len(self.dev_samples) < config.num_samples:
                self.dev_samples.extend(self.pipeline_pool.run())
            self.dev_samples = self.dev_samples[:config.num_samples]
            self.pipeline_pool.cleanup()

    def __iter__(self):
        if self.config.is_dev:
            assert len(self.dev_samples) == self.config.num_samples
            yield from self.dev_samples
            return

        cached_samples: List[Sample] = []

        for _ in range(self.config.num_samples):

            while not cached_samples:
                if not self.config.num_cached_runs:
                    cached_samples.extend(self.pipeline_pool.run())

                else:
                    cur_cached_samples: List[Sample] = []
                    for _ in range(self.config.num_cached_runs):
                        cur_cached_samples.extend(self.pipeline_pool.run())
                    shuffled_indices = list(range(len(cur_cached_samples)))
                    self.rng.shuffle(shuffled_indices)
                    for idx in shuffled_indices:
                        cached_samples.append(cur_cached_samples[idx])

                if not cached_samples:
                    logger.warning('cached_samples not filled!')

            yield cached_samples.pop()


def adaptive_scaling_dataset_collate_fn(batch: Iterable[Sample]):
    default_batch: List[Dict[str, np.ndarray]] = []

    downsampled_shape = None
    downsampled_core_box = None

    for (
        image,
        downsampled_shape,
        downsampled_core_box,
        downsampled_mask,
        downsampled_score_map,
    ) in batch:
        default_batch.append({
            # (H, W, 3) -> (3, H, W).
            'image': np.transpose(image.mat, axes=(2, 0, 1)).astype(np.float32),
            'downsampled_mask': downsampled_mask.np_mask.astype(np.float32),
            'downsampled_score_map': downsampled_score_map.mat,
        })
        downsampled_shape = downsampled_shape
        downsampled_core_box = downsampled_core_box

    assert downsampled_shape and downsampled_core_box
    collated_batch: Dict[str, Any] = default_collate(default_batch)
    collated_batch['downsampled_shape'] = downsampled_shape
    collated_batch['downsampled_core_box'] = downsampled_core_box

    return collated_batch
