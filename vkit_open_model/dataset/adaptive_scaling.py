from typing import Tuple, Optional, Iterable, Dict, List, Any, Sequence
import logging

import numpy as np
from numpy.random import Generator as RandomGenerator
from torch.utils.data import IterableDataset, default_collate

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


class AdaptiveScalingIterableDataset(IterableDataset):

    def __init__(
        self,
        steps_json: PathType,
        num_samples: int,
        rng_seed: int,
        num_processes: int,
        num_samples_reset_rng: Optional[int] = None,
        is_dev: bool = False,
    ):
        super().__init__()

        logger.info('Creating pipeline pool...')
        num_runs_reset_rng = None
        if num_samples_reset_rng:
            assert num_samples_reset_rng % num_processes == 0
            num_runs_reset_rng = num_samples_reset_rng // num_processes

        self.pipeline_pool = PipelinePool(
            pipeline=Pipeline(
                steps=pipeline_step_collection_factory.create(steps_json),
                post_processor=adaptive_scaling_pipeline_post_processor_factory.create(),
            ),
            inventory=num_processes * 7,
            rng_seed=rng_seed,
            num_processes=num_processes,
            num_runs_reset_rng=num_runs_reset_rng,
        )
        logger.info('Pipeline pool created.')

        self.num_samples = num_samples
        self.is_dev = is_dev

        self.dev_samples: List[Sample] = []
        if self.is_dev:
            while len(self.dev_samples) < self.num_samples:
                self.dev_samples.extend(self.pipeline_pool.run())
            self.dev_samples = self.dev_samples[:self.num_samples]
            self.pipeline_pool.cleanup()

    def __iter__(self):
        if self.is_dev:
            assert len(self.dev_samples) == self.num_samples
            yield from self.dev_samples
            return

        cached_samples: List[Sample] = []

        for _ in range(self.num_samples):
            if not cached_samples:
                cached_samples.extend(self.pipeline_pool.run())
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
