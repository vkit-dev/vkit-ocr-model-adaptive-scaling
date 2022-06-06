import torch
from vkit_open_model.model.upernext import UperNext


def test_upernext():
    model = UperNext.create_tiny(100)
    features = [
        torch.rand(1, 96, 80, 80),
        torch.rand(1, 192, 40, 40),
        torch.rand(1, 384, 20, 20),
        torch.rand(1, 768, 10, 10),
    ]
    output = model(features)
    assert output.shape == (1, 100, 80, 80)


def test_upernext_jit():
    model = UperNext.create_tiny(100)
    model_jit = torch.jit.script(model)  # type: ignore
    assert model_jit
