from typing import Tuple, Generator, Optional, Iterable, Dict, List, Any, Union, Sequence
from collections import abc
import logging

import numpy as np
from numpy.random import RandomState
from torch.utils.data import IterableDataset, default_collate, get_worker_info

from vkit.element import Image, Mask, ScoreMap, Box
from vkit.utility import PathType
from vkit.pipeline import (
    pipeline_step_collection_factory,
    PipelineState,
    PageCroppingStep,
    NoneTypePipelinePostProcessorConfig,
    PipelinePostProcessor,
    PipelinePostProcessorFactory,
    Pipeline,
)

logger = logging.getLogger(__name__)

Sample = Tuple[Image, Tuple[int, int], Box, Mask, ScoreMap]


class AdaptiveScalingPipelinePostProcessor(
    PipelinePostProcessor[
        NoneTypePipelinePostProcessorConfig,
        Generator[Sample, None, None],
    ]
):  # yapf: disable

    def generate_output(self, state: PipelineState, rnd: RandomState):
        page_cropping_step_output = state.get_pipeline_step_output(PageCroppingStep)
        for cropped_page in page_cropping_step_output.cropped_pages:
            downsampled_label = cropped_page.downsampled_label
            assert downsampled_label
            yield (
                cropped_page.page_image,
                downsampled_label.shape,
                downsampled_label.core_box,
                downsampled_label.page_text_line_mask,
                downsampled_label.page_text_line_height_score_map,
            )


adaptive_scaling_pipeline_post_processor_factory = PipelinePostProcessorFactory(
    AdaptiveScalingPipelinePostProcessor
)


class AdaptiveScalingIterableDataset(IterableDataset):

    def __init__(
        self,
        steps_json: PathType,
        num_steps: int,
        rnd_seed: Optional[Union[int, Sequence[int]]] = None,
    ):
        super().__init__()

        logger.info('Creating pipeline...')
        self.pipeline = Pipeline(
            steps=pipeline_step_collection_factory.create(steps_json),
            post_processor=adaptive_scaling_pipeline_post_processor_factory.create(),
        )
        logger.info('Pipeline created...')

        self.num_steps = num_steps
        self.rnd_seed = rnd_seed
        if isinstance(self.rnd_seed, abc.Sequence):
            assert len(self.rnd_seed) == self.num_steps

    def get_rnd(self, rnd: Optional[RandomState], rnd_seed_seq_idx: int):
        if self.rnd_seed is None or isinstance(self.rnd_seed, int):
            if rnd is None:
                rnd = RandomState(self.rnd_seed)
        else:
            rnd = RandomState(self.rnd_seed[rnd_seed_seq_idx])
            rnd_seed_seq_idx = (rnd_seed_seq_idx + 1) % len(self.rnd_seed)
        return rnd, rnd_seed_seq_idx

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            num_steps = self.num_steps
            rnd_seed_seq_idx = 0
        else:
            num_steps = self.num_steps // worker_info.num_workers
            rnd_seed_seq_idx = worker_info.id * num_steps

        rnd = None
        while num_steps > 0:
            rnd, rnd_seed_seq_idx = self.get_rnd(rnd, rnd_seed_seq_idx)
            try:
                for sample in self.pipeline.run(rnd):
                    yield sample
                    num_steps -= 1
                    if num_steps <= 0:
                        break
            except Exception:
                logger.exception('pipeline failed, retrying...')


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
            'image': image.mat,
            'downsampled_mask': downsampled_mask.mat,
            'downsampled_score_map': downsampled_score_map.mat,
        })
        downsampled_shape = downsampled_shape
        downsampled_core_box = downsampled_core_box

    assert downsampled_shape and downsampled_core_box
    collated_batch: Dict[str, Any] = default_collate(default_batch)
    collated_batch['downsampled_shape'] = downsampled_shape
    collated_batch['downsampled_core_box'] = downsampled_core_box

    return collated_batch
