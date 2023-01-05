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
from vkit_open_model.model.pan import (
    PanNeckTopDownTopBlock,
    PanNeckTopDownNormalBlock,
    PanNeckBottomUpBlock,
    PanNeck,
    PanHead,
)


def test_pan_top_down_head_block():
    block = PanNeckTopDownTopBlock(upper_channels=256, lower_channels=128)
    model_jit = torch.jit.script(block)  # type: ignore
    assert model_jit

    x0, x1 = model_jit(torch.rand(1, 256, 10, 10))  # type: ignore
    assert x0.shape == (1, 128, 10, 10)
    assert x1.shape == (1, 128, 10, 10)
    assert (x0 == x1).sum() == 0


def test_pan_top_down_normal_block():
    block = PanNeckTopDownNormalBlock(upper_channels=256, lower_channels=128, is_bottom=False)
    model_jit = torch.jit.script(block)  # type: ignore
    assert model_jit

    x0, x1 = model_jit(torch.rand(1, 256, 5, 5), torch.rand(1, 256, 10, 10))  # type: ignore
    assert x0.shape == (1, 128, 10, 10)
    assert x1.shape == (1, 128, 10, 10)
    assert (x0 == x1).sum() == 0


def test_pan_neck_bottom_up_block():
    block = PanNeckBottomUpBlock(lower_channels=128, upper_channels=256)
    model_jit = torch.jit.script(block)  # type: ignore
    assert model_jit

    output = model_jit(torch.rand(1, 128, 10, 10), torch.rand(1, 128, 5, 5))  # type: ignore
    assert output.shape == (1, 256, 5, 5)


def test_pan_neck():
    neck = PanNeck(in_channels_group=(96, 192, 384, 768))
    features = [
        torch.rand(1, 96, 80, 80),
        torch.rand(1, 192, 40, 40),
        torch.rand(1, 384, 20, 20),
        torch.rand(1, 768, 10, 10),
    ]
    outputs = neck(features)
    assert len(outputs) == len(features)
    for output, feature in zip(outputs, features):
        assert output.shape == feature.shape

    model_jit = torch.jit.script(neck)  # type: ignore
    assert model_jit
    del neck

    outputs = model_jit(features)  # type: ignore
    assert len(outputs) == len(features)
    for output, feature in zip(outputs, features):
        assert output.shape == feature.shape
    del model_jit


def test_pan_head():
    head = PanHead(in_channels=192, out_channels=4, init_output_bias=10.0)
    output = head(torch.rand(1, 192, 40, 40))
    assert output.shape == (1, 4, 40, 40)
    assert 9.0 <= output.mean() <= 11.0

    model_jit = torch.jit.script(head)  # type: ignore
    assert model_jit
    del model_jit, head
