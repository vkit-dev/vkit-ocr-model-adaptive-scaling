from typing import Sequence, Optional

import torch
from torch import nn


def conv1x1(in_channels: int, out_channels: int):
    return nn.Linear(
        in_features=in_channels,
        out_features=out_channels,
    )


def conv3x3(in_channels: int, out_channels: int):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1,
    )


def pconv2x2(in_channels: int, out_channels: int):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        stride=2,
    )


def pconv4x4(in_channels: int, out_channels: int):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=4,
        stride=4,
    )


def dconv7x7(in_channels: int, out_channels: Optional[int] = None):
    if out_channels is None:
        out_channels = in_channels
    else:
        assert in_channels % out_channels == 0

    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        padding=3,
        groups=in_channels,
    )


class Permutation(nn.Module):

    def __init__(self, dims: Sequence[int]):
        super().__init__()
        self.dims = tuple(dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return torch.permute(x, self.dims)


def permute_bchw_to_bhwc():
    # `(B, C, H, W)` -> `(B, H, W, C)`
    return Permutation([0, 2, 3, 1])


def permute_bhwc_to_bchw():
    # `(B, H, W, C)` -> `(B, C, H, W)`
    return Permutation([0, 3, 1, 2])


def ln(in_channels: int):
    return nn.LayerNorm(in_channels, eps=1E-6)


def gelu():
    return nn.GELU()
