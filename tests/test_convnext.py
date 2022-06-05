import torch
from vkit_open_model.model.convnext import ConvNext


def print_resnet50():
    from torchvision.models.resnet import resnet50
    model = resnet50()
    print(model)


def print_convnext_tiny():
    from torchvision.models.convnext import convnext_tiny
    model = convnext_tiny()
    print(model.features[2])


def test_convnext():
    model = ConvNext.create_tiny()
    model = torch.jit.script(model)  # type: ignore
    x = torch.rand((1, 3, 320, 320))
    features = model(x)  # type: ignore
    assert len(features) == 4
    assert features[0].shape == (1, 96, 80, 80)
    assert features[1].shape == (1, 192, 40, 40)
    assert features[2].shape == (1, 384, 20, 20)
    assert features[3].shape == (1, 768, 10, 10)
