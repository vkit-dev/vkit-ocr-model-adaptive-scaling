from vkit_open_model.model.adaptive_scaling import AdaptiveScaling
import torch


def test_adaptive_scaling_jit():
    model = AdaptiveScaling.create_tiny(stem_use_pconv2x2=True)
    model_jit = torch.jit.script(model)  # type: ignore

    x = torch.rand((1, 3, 320, 320))
    mask_feature, scale_feature = model_jit(x)  # type: ignore
    assert mask_feature.shape == (1, 1, 160, 160)
    assert scale_feature.shape == (1, 1, 160, 160)
