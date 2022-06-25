from typing import Tuple, Generator, Optional, Iterable, Dict, List, Any
import logging

import numpy as np
from numpy.random import RandomState
from torch.utils.data import IterableDataset, default_collate

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
        rnd_seed: Optional[int] = None,
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

    def __iter__(self):
        rnd = RandomState(self.rnd_seed)
        num_steps = self.num_steps
        while num_steps > 0:
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
