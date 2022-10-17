# This project (vkit-x/vkit-open-model) is dual-licensed under commercial and SSPL licenses.
#
# The commercial license gives you the full rights to create and distribute software
# on your own terms without any SSPL license obligations. For more information,
# please see the "LICENSE_COMMERCIAL.txt" file.
#
# This project is also available under Server Side Public License (SSPL).
# The SSPL licensing is ideal for use cases such as open source projects with
# SSPL distribution, student/academic purposes, hobby projects, internal research
# projects without external distribution, or other projects where all SSPL
# obligations can be met. For more information, please see the "LICENSE_SSPL.txt" file.
from typing import Dict, Any
import random

import torch
import numpy as np


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
    batch_size: int,
    num_batches: int,
):
    num_samples = batch_size * num_batches
    return num_samples
