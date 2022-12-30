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
from vkit_open_model.model import PanNeck


def test_pan():
    neck = PanNeck(
        in_channels_group=(96, 192, 384, 768),
        out_channels=384,
    )
    features = [
        torch.rand(1, 96, 80, 80),
        torch.rand(1, 192, 40, 40),
        torch.rand(1, 384, 20, 20),
        torch.rand(1, 768, 10, 10),
    ]
    outputs = neck(features)
    assert len(outputs) == 4
    assert outputs[0].shape == (1, 384, 80, 80)
    assert outputs[1].shape == (1, 384, 40, 40)
    assert outputs[2].shape == (1, 384, 20, 20)
    assert outputs[3].shape == (1, 384, 10, 10)

    model_jit = torch.jit.script(neck)  # type: ignore
    assert model_jit
