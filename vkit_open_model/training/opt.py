from typing import Dict, Set, Any
import random

import torch
import numpy as np
from numpy.random import default_rng


def batch_to_device(batch: Dict[str, Any], device: torch.device):
    return {
        key: val.to(device, non_blocking=True) if torch.is_tensor(val) else val
        for key, val in batch.items()
    }


def device_is_cuda(device: torch.device):
    return (device.type == 'cuda')


def enable_cudnn_benchmark(device: torch.device):
    if device_is_cuda(device):
        torch.backends.cudnn.benchmark = True  # type: ignore


def enable_cudnn_deterministic(device: torch.device):
    if device_is_cuda(device):
        torch.backends.cudnn.deterministic = True  # type: ignore


def setup_seeds(
    random_seed: int = 13370,
    numpy_seed: int = 1337,
    torch_seed: int = 133,
):
    random.seed(random_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)


def calculate_iterable_dataset_num_samples(
    num_workers: int,
    batch_size: int,
    num_batches: int,
):
    num_samples = batch_size * num_batches
    if num_workers > 0:
        assert num_samples % num_workers == 0
    return num_samples


def generate_iterable_dataset_rng_seeds(num_samples: int, rng_seed: int):
    rng = default_rng(rng_seed)

    rng_seeds_set: Set[int] = set()
    while len(rng_seeds_set) < num_samples:
        seed: int = rng.bit_generator.state['state']['state']
        rng_seeds_set.add(seed)
        rng.random()

    rng_seeds = list(rng_seeds_set)
    rng.shuffle(rng_seeds)
    assert len(rng_seeds) == num_samples

    return rng_seeds
