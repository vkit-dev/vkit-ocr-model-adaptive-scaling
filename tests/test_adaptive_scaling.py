import torch
from torch.utils.data import DataLoader

from vkit_open_model.model.adaptive_scaling import AdaptiveScaling
from vkit_open_model.dataset.adaptive_scaling import (
    adaptive_scaling_dataset_collate_fn,
    AdaptiveScalingIterableDataset,
)


def test_adaptive_scaling_jit():
    model = AdaptiveScaling.create_tiny()
    model_jit = torch.jit.script(model)  # type: ignore

    x = torch.rand((1, 3, 320, 320))
    mask_feature, scale_feature = model_jit(x)  # type: ignore
    assert mask_feature.shape == (1, 1, 160, 160)
    assert scale_feature.shape == (1, 1, 160, 160)


def debug_adaptive_scaling_dataset():
    data_loader = DataLoader(
        dataset=AdaptiveScalingIterableDataset(
            steps_json='$VKIT_ARTIFACT_PACK/pipeline/text_detection/adaptive_scaling.json',
            num_steps=20,
        ),
        batch_size=3,
        num_workers=2,
        collate_fn=adaptive_scaling_dataset_collate_fn,
    )
    data_loader_it = iter(data_loader)
    batch = next(data_loader_it)
    assert batch
    breakpoint()
