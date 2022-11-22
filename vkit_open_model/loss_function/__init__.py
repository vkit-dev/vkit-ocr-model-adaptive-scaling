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
from .weighted_bce_with_logits import WeightedBceWithLogitsLossFunction
from .cross_entropy_with_logits import CrossEntropyWithLogitsLossFunction
from .focal_with_logits import FocalWithLogitsLossFunction
from .l1 import L1LossFunction
from .l2 import L2LossFunction
from .weight_adaptive_heatmap_regression import WeightAdaptiveHeatmapRegressionLossFunction
from .dice import DiceLossFunction
from .adaptive_scaling import (
    AdaptiveScalingRoughLossFunctionConifg,
    AdaptiveScalingRoughLossFunction,
    AdaptiveScalingPreciseLossFunctionConifg,
    AdaptiveScalingPreciseLossFunction,
)
