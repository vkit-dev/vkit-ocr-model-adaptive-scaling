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
from vkit_open_model.model.upernext import UperNextNeck


def test_upernext():
    model = UperNextNeck(
        in_channels_group=(96, 192, 384, 768),
        out_channels=384,
    )
    features = [
        torch.rand(1, 96, 80, 80),
        torch.rand(1, 192, 40, 40),
        torch.rand(1, 384, 20, 20),
        torch.rand(1, 768, 10, 10),
    ]
    output = model(features)
    assert output.shape == (1, 384, 80, 80)

    model_jit = torch.jit.script(model)  # type: ignore
    assert model_jit
