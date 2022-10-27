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
from vkit_open_model.inferencing.opt import pad_length_to_make_divisible


def test_pad_length_to_make_divisible():
    padded_length, pad = pad_length_to_make_divisible(6, 3)
    assert padded_length == 6
    assert pad == 0

    padded_length, pad = pad_length_to_make_divisible(7, 3)
    assert padded_length == 9
    assert pad == 2
