from typing import Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        normalization = nn.InstanceNorm2d,
        activation = partial(nn.ReLU, inplace=True),
    ) -> None:
        super().__init__()

        padding = ((kernel_size - 1) * dilation) // 2
        use_bias = (normalization == nn.Identity)

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                use_bias),
            normalization(out_channels),
            activation(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

