from typing import List, Optional, Union, Sequence
from enum import Enum, unique
import math

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from numpy.random import default_rng

from vkit_open_model.train import Metrics, SecondOrderRandomGenerator, setup_seeds


@unique
class DebugMetricsTag(Enum):
    FOO = 'foo'
    BAR = 'bar'


def test_metrics():
    metrics = Metrics(DebugMetricsTag, 3)
    assert math.isclose(metrics.update(DebugMetricsTag.FOO, 1), 1)
    assert math.isclose(metrics.update(DebugMetricsTag.FOO, 2), 1.5)
    assert math.isclose(metrics.update(DebugMetricsTag.FOO, 3), 2)
    assert math.isclose(metrics.update(DebugMetricsTag.FOO, 4), 3)


class TestIterableDataset(IterableDataset):

    def __init__(self):
        super().__init__()
        self.epoch_idx = 0

    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info
        seed = worker_info.seed  # type: ignore
        rng = default_rng(seed)
        for _ in range(self.epoch_idx):
            rng.random()

        for _ in range(5):
            yield {'num': rng.integers(0, 1024)}

        self.epoch_idx += 1


def test_rng_seeding():
    setup_seeds(torch_seed=42)

    test_data_loader = DataLoader(
        TestIterableDataset(),
        batch_size=5,
        num_workers=2,
        persistent_workers=True,
    )

    batches: List[torch.Tensor] = []

    for epoch_idx in range(2):
        for batch_idx, batch in enumerate(test_data_loader):
            print(f'epoch_idx={epoch_idx}, batch_idx={batch_idx}, batch={batch}')
            batches.append(batch['num'])

    assert len(batches) == 4
    assert batches[0].tolist() == [709, 884, 651, 745, 790]
    assert batches[1].tolist() == [600, 986, 71, 882, 139]
    assert batches[2].tolist() == [651, 745, 790, 1002, 863]
    assert batches[3].tolist() == [71, 882, 139, 145, 605]


class TestSecondOrderRandomGeneratorIterableDataset(IterableDataset):

    def __init__(
        self,
        num_samples: int,
        rng_seed: Optional[Union[int, Sequence[int]]] = None,
    ):
        super().__init__()

        self.epoch_idx = 0
        self.second_order_rng = SecondOrderRandomGenerator(
            rng_seed=rng_seed,
            num_samples=num_samples,
        )

    def __iter__(self):
        for rng in self.second_order_rng.get_rngs(epoch_idx=self.epoch_idx):
            yield {'num': rng.integers(0, 1024)}
        self.epoch_idx += 1


def test_second_order_rng():
    setup_seeds(torch_seed=42)

    test_data_loader = DataLoader(
        TestSecondOrderRandomGeneratorIterableDataset(num_samples=10),
        batch_size=5,
        num_workers=2,
        persistent_workers=True,
    )

    batches: List[torch.Tensor] = []

    for epoch_idx in range(2):
        for batch_idx, batch in enumerate(test_data_loader):
            print(f'epoch_idx={epoch_idx}, batch_idx={batch_idx}, batch={batch}')
            batches.append(batch['num'])

    assert len(batches) == 4
    assert batches[0].tolist() == [935, 442, 279, 764, 740]
    assert batches[1].tolist() == [280, 0, 1021, 185, 589]
    assert batches[2].tolist() == [980, 722, 266, 599, 95]
    assert batches[3].tolist() == [372, 820, 677, 442, 198]
