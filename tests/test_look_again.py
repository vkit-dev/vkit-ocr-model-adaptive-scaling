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

from vkit_open_model.model import LookAgainConfig, LookAgain


def test_look_again():
    model = LookAgain(LookAgainConfig())

    print('forward_rough')
    x = torch.rand(1, 3, 320, 320)
    rough_classification_logits, rough_char_scale_logits = model.forward_rough(x)
    assert rough_classification_logits.shape == (1, 3, 80, 80)
    assert rough_char_scale_logits.shape == (1, 1, 80, 80)

    print('forward_precise')
    (
        precise_char_localization_logits_group,
        precise_char_objectness_logits_group,
        precise_char_orientation_logits_group,
    ) = model.forward_precise(x)
    assert len(precise_char_localization_logits_group) \
        == len(precise_char_objectness_logits_group) \
        == len(precise_char_orientation_logits_group) \
        == 3
    assert precise_char_localization_logits_group[0].shape == (1, 4, 40, 40)
    assert precise_char_objectness_logits_group[0].shape == (1, 1, 40, 40)
    assert precise_char_orientation_logits_group[0].shape == (1, 4, 40, 40)
    assert precise_char_localization_logits_group[1].shape == (1, 4, 20, 20)
    assert precise_char_objectness_logits_group[1].shape == (1, 1, 20, 20)
    assert precise_char_orientation_logits_group[1].shape == (1, 4, 20, 20)
    assert precise_char_localization_logits_group[2].shape == (1, 4, 10, 10)
    assert precise_char_objectness_logits_group[2].shape == (1, 1, 10, 10)
    assert precise_char_orientation_logits_group[2].shape == (1, 4, 10, 10)

    print('jit forward_rough')
    model_jit = torch.jit.script(model)  # type: ignore
    del model
    rough_classification_logits, rough_char_scale_logits = \
        model_jit.forward_rough(x)  # type: ignore
    assert rough_classification_logits.shape == (1, 3, 80, 80)
    assert rough_char_scale_logits.shape == (1, 1, 80, 80)

    print('jit forward_precise')
    (
        precise_char_localization_logits_group,
        precise_char_objectness_logits_group,
        precise_char_orientation_logits_group,
    ) = model_jit.forward_precise(x)  # type: ignore
    assert len(precise_char_localization_logits_group) \
        == len(precise_char_objectness_logits_group) \
        == len(precise_char_orientation_logits_group) \
        == 3
    assert precise_char_localization_logits_group[0].shape == (1, 4, 40, 40)
    assert precise_char_objectness_logits_group[0].shape == (1, 1, 40, 40)
    assert precise_char_orientation_logits_group[0].shape == (1, 4, 40, 40)
    assert precise_char_localization_logits_group[1].shape == (1, 4, 20, 20)
    assert precise_char_objectness_logits_group[1].shape == (1, 1, 20, 20)
    assert precise_char_orientation_logits_group[1].shape == (1, 4, 20, 20)
    assert precise_char_localization_logits_group[2].shape == (1, 4, 10, 10)
    assert precise_char_objectness_logits_group[2].shape == (1, 1, 10, 10)
    assert precise_char_orientation_logits_group[2].shape == (1, 4, 10, 10)
