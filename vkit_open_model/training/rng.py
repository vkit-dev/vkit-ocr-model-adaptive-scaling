from typing import cast, Optional, Union, Sequence
from collections import abc
import logging

from torch.utils.data import get_worker_info
from numpy.random import default_rng, Generator as RandomGenerator

logger = logging.getLogger(__name__)


class SecondOrderRandomGenerator:

    def __init__(
        self,
        rng_seed: Optional[Union[int, Sequence[int]]],
        num_samples: int,
    ):
        self.rng_seed = rng_seed
        self.num_samples = num_samples
        if isinstance(self.rng_seed, abc.Sequence):
            assert len(self.rng_seed) == self.num_samples

    def get_num_samples(self):
        worker_info = get_worker_info()
        if worker_info:
            worker_num_samples = self.num_samples // worker_info.num_workers
            return worker_num_samples
        else:
            return self.num_samples

    def get_next_rng(
        self,
        epoch_idx: int,
        sample_idx: int,
        rng: Optional[RandomGenerator],
    ):
        if isinstance(self.rng_seed, abc.Sequence):
            worker_info = get_worker_info()
            if worker_info:
                worker_idx: int = worker_info.id
                worker_num_samples = self.num_samples // worker_info.num_workers
                rng_seed_seq_begin = worker_num_samples * worker_idx
                rng_seed_seq_idx = rng_seed_seq_begin + sample_idx
            else:
                rng_seed_seq_idx = sample_idx

            next_rng = default_rng(self.rng_seed[rng_seed_seq_idx])

        else:
            if not rng:
                rng_seed = self.rng_seed

                if not rng_seed:
                    # Only for num_workers > 0.
                    worker_info = get_worker_info()
                    assert worker_info
                    worker_seed: int = worker_info.seed

                    init_rng = default_rng(worker_seed)
                    # Initialize seed for epoch.
                    for _ in range(epoch_idx):
                        init_rng.random()
                    rng_seed = cast(int, init_rng.bit_generator.state['state']['state'])
                    logger.info(f'Initialize rng_seed={rng_seed} for worker_idx={worker_info.id}')

                next_rng = default_rng(rng_seed)
            else:
                next_rng = rng

        return next_rng

    def get_rngs(self, epoch_idx: int):
        rng = None
        for sample_idx in range(self.get_num_samples()):
            rng = self.get_next_rng(
                epoch_idx=epoch_idx,
                sample_idx=sample_idx,
                rng=rng,
            )
            yield rng
