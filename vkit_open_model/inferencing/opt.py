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
import math
import numpy as np


def pad_length_to_make_divisible(length: int, downsampling_factor: int):
    padded_length = math.ceil(length / downsampling_factor) * downsampling_factor
    return padded_length, padded_length - length


def pad_mat_to_make_divisible(
    # (H, W, *).
    mat: np.ndarray,
    downsampling_factor: int,
):
    height, width = mat.shape[:2]
    height, height_pad = pad_length_to_make_divisible(height, downsampling_factor)
    width, width_pad = pad_length_to_make_divisible(width, downsampling_factor)

    if height_pad == 0 and width_pad == 0:
        # No need to pad.
        return mat

    padded_shape = list(mat.shape)
    padded_shape[0] = height
    padded_shape[1] = width

    padded_mat = np.zeros(padded_shape, dtype=mat.dtype)
    padded_mat[:height - height_pad, :width - width_pad] = mat

    return padded_mat
