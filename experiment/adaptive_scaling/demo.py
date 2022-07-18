import torch
import numpy as np
import cv2 as cv
import iolite as io

from vkit.element import Image, Mask, ScoreMap, Painter
from vkit_open_model.evaluation import pad_mat_to_make_divisible


def infer(
    model_jit_path: str,
    image_file: str,
    output_folder: str,
    downsample_short_side_legnth: int = 640,
):
    model_jit: torch.jit.ScriptModule = torch.jit.load(model_jit_path)  # type: ignore
    model_jit = model_jit.eval()
    torch.set_grad_enabled(False)

    image = Image.from_file(image_file).to_rgb_image()
    if min(image.height, image.width) > downsample_short_side_legnth:
        if image.height < image.width:
            image = image.to_resized_image(
                resized_height=downsample_short_side_legnth,
                cv_resize_interpolation=cv.INTER_AREA,
            )
        else:
            image = image.to_resized_image(
                resized_width=downsample_short_side_legnth,
                cv_resize_interpolation=cv.INTER_AREA,
            )

    mat = pad_mat_to_make_divisible(image.mat, downsampling_factor=32)
    image = Image(mat=mat)

    mat = np.transpose(mat, axes=(2, 0, 1)).astype(np.float32)
    x = torch.from_numpy(mat).unsqueeze(0)
    with torch.no_grad():
        mask_feature, scale_feature = model_jit(x)  # type: ignore

    mask_mat = (torch.sigmoid(mask_feature[0][0]) >= 0.5).numpy().astype(np.uint8)
    mask = Mask(mat=mask_mat)
    assert mask.height * 2 == image.height
    assert mask.width * 2 == image.width
    mask = mask.to_resized_mask(
        resized_height=image.height,
        resized_width=image.width,
        cv_resize_interpolation=cv.INTER_NEAREST,
    )

    score_map_mat = scale_feature[0][0].numpy().astype(np.float32)
    score_map = ScoreMap(mat=score_map_mat, is_prob=False)
    assert score_map.height * 2 == image.height
    assert score_map.width * 2 == image.width
    score_map = score_map.to_resized_score_map(
        resized_height=image.height,
        resized_width=image.width,
        cv_resize_interpolation=cv.INTER_NEAREST,
    )

    out_fd = io.folder(output_folder, touch=True)
    painter = Painter(image.copy())
    painter.paint_mask(mask)
    painter.to_file(out_fd / 'mask.png')

    painter = Painter(image.copy())
    painter.paint_score_map(score_map, alpha=1.0)
    painter.to_file(out_fd / 'score_map.png')
