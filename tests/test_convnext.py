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
import torch
import iolite as io
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
    x = torch.rand((1, 3, 320, 320))
    features = model(x)  # type: ignore
    assert len(features) == 4
    assert features[0].shape == (1, 96, 80, 80)
    assert features[1].shape == (1, 192, 40, 40)
    assert features[2].shape == (1, 384, 20, 20)
    assert features[3].shape == (1, 768, 10, 10)


def test_convnext_jit():
    model = ConvNext.create_tiny(stem_use_pconv2x2=True)
    model_jit = torch.jit.script(model)  # type: ignore

    x = torch.rand((1, 3, 320, 320))
    features = model_jit(x)  # type: ignore
    assert len(features) == 4
    assert features[0].shape == (1, 96, 160, 160)
    assert features[1].shape == (1, 192, 80, 80)
    assert features[2].shape == (1, 384, 40, 40)
    assert features[3].shape == (1, 768, 20, 20)

    out_fd = io.folder(
        '$VKIT_OPEN_MODEL_DATA/test_convnext',
        expandvars=True,
        touch=True,
    )
    torch.save(
        {'model': model_jit.state_dict()},  # type: ignore
        out_fd / 'torch-save-convnext-state-dict.pt',
    )

    dump = torch.load(out_fd / 'torch-save-convnext-state-dict.pt')
    model.load_state_dict(dump['model'])
