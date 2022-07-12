from typing import Tuple, Generator, Optional, Iterable, Dict, List, Any, Union, Sequence
import logging

import numpy as np
from numpy.random import Generator as RandomGenerator
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
from vkit_open_model.training import SecondOrderRandomGenerator

logger = logging.getLogger(__name__)

Sample = Tuple[Image, Tuple[int, int], Box, Mask, ScoreMap]


class AdaptiveScalingPipelinePostProcessor(
    PipelinePostProcessor[
        NoneTypePipelinePostProcessorConfig,
        Generator[Sample, None, None],
    ]
):  # yapf: disable

    def generate_output(self, state: PipelineState, rng: RandomGenerator):
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
        num_samples: int,
        rng_seed: Optional[Union[int, Sequence[int]]] = None,
    ):
        super().__init__()

        logger.info('Creating pipeline...')
        self.pipeline = Pipeline(
            steps=pipeline_step_collection_factory.create(steps_json),
            post_processor=adaptive_scaling_pipeline_post_processor_factory.create(),
        )
        logger.info('Pipeline created.')

        self.epoch_idx = 0
        self.second_order_rng = SecondOrderRandomGenerator(
            rng_seed=rng_seed,
            num_samples=num_samples,
        )

    def __iter__(self):
        samples_queue: List[Sample] = []

        for rng in self.second_order_rng.get_rngs(epoch_idx=self.epoch_idx):
            if not samples_queue:
                # Generate new samples based on rng.
                while True:
                    try:
                        samples_queue.extend(self.pipeline.run(rng))
                        break
                    except Exception:
                        logger.exception('pipeline failed. retrying...')
                        # Force new rng.
                        rng.random()

            assert samples_queue
            yield samples_queue.pop()

        self.epoch_idx += 1


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
