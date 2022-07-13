from typing import Optional, Union, Sequence, Mapping, Any
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

        self.rng_state: Optional[Mapping[str, Any]] = None

    def get_num_samples(self):
        worker_info = get_worker_info()
        if worker_info:
            worker_num_samples = self.num_samples // worker_info.num_workers
            return worker_num_samples
        else:
            return self.num_samples

    def get_next_rng(
        self,
        rng: Optional[RandomGenerator],
        sample_idx: Optional[int] = None,
    ):
        if isinstance(self.rng_seed, abc.Sequence):
            assert sample_idx is not None
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
            if rng is None:
                if self.rng_seed is not None:
                    next_rng = default_rng(self.rng_seed)

                else:
                    # Only for num_workers > 0.
                    worker_info = get_worker_info()
                    assert worker_info

                    if self.rng_state is None:
                        worker_seed: int = worker_info.seed
                        next_rng = default_rng(worker_seed)
                    else:
                        next_rng = default_rng()
                        next_rng.bit_generator.state = self.rng_state

                    logger.info(
                        f'Initialize rng_state={next_rng.bit_generator.state} '
                        f'for worker_idx={worker_info.id}'
                    )

            else:
                next_rng = rng

        return next_rng

    def get_rngs(self):
        rng = None

        for sample_idx in range(self.get_num_samples()):
            rng = self.get_next_rng(
                sample_idx=sample_idx,
                rng=rng,
            )
            yield rng

        assert rng is not None
        self.rng_state = rng.bit_generator.state
