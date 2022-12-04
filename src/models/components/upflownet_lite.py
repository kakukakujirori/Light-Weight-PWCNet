from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.basic_conv import ConvNormAct
from src.models.components.flow_utils import (
    backwarp,
    normalize_features,
    upsample2d_flow,
)


class FeatureExtractor(nn.Module):
    def __init__(self, num_chs: list[int], if_end_relu: bool = True, if_end_norm: bool = False):
        super().__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()
        for ch_in, ch_out in zip(num_chs[:-1], num_chs[1:]):
            self.convs.append(
                nn.Sequential(
                    ConvNormAct(ch_in, ch_out, stride=2),
                    ConvNormAct(
                        ch_out,
                        ch_out,
                        normalization=nn.InstanceNorm2d if if_end_norm else nn.Identity,
                        activation=partial(nn.ReLU, inplace=True) if if_end_relu else nn.Identity,
                    ),
                )
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feature_pyramid = [x]
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid


class SGUModel(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.dense_block = nn.Sequential(
            ConvNormAct(feature_dim * 2, feature_dim),
            ConvNormAct(feature_dim, feature_dim // 2),
            ConvNormAct(feature_dim // 2, 3, normalization=nn.Identity),
        )

    def forward(self, flow_init: torch.Tensor, feature1: torch.Tensor, feature2: torch.Tensor):
        assert flow_init.shape[-2:] == feature1.shape[-2:] == feature2.shape[-2:]
        feature2_warped = backwarp(feature2, flow_init)

        out = self.dense_block(torch.cat([feature1, feature2_warped], dim=1))
        inter_flow = out[:, 0:2, :, :]
        inter_mask = torch.sigmoid(out[:, 2:3, :, :])

        return backwarp(flow_init, inter_flow) * (1 - inter_mask) + flow_init * inter_mask


class Correlation(nn.Module):
    def __init__(self, kernel_size: int = 1, max_displacement: int = 4):
        super().__init__()
        self.pad_size = max_displacement
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        self.padlayer = nn.ConstantPad2d(max_displacement, 0)

    def forward(self, in1: torch.Tensor, in2: torch.Tensor) -> torch.Tensor:
        bz, _, hei, wid = in1.shape
        f1 = self.unfold(in1)
        f2 = self.unfold(in2)

        searching_kernel_size = f2.shape[1]
        f2_ = f2.reshape(-1, 1, hei, wid)
        f2 = F.unfold(f2_, kernel_size=(hei, wid), padding=self.pad_size, stride=1)

        _, kernel_number, window_number = f2.shape
        f2_ = f2.reshape(bz, searching_kernel_size, kernel_number, window_number)
        f2_2 = f2_.permute(0, 3, 1, 2)
        f1_2 = f1.unsqueeze(1)
        res = f2_2 * f1_2
        res = torch.mean(res, dim=2)
        res = res.reshape(bz, window_number, hei, wid)
        return res


class FlowEstimator(nn.Module):
    def __init__(
        self,
        in_ch: int,
        f_channels: list[int] = (128, 64, 32),
        out_ch: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        pre_ch = in_ch
        for ch in f_channels:
            self.layers.append(ConvNormAct(pre_ch, ch, 3))
            pre_ch = ch

        self.conv_last = nn.Conv2d(pre_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for conv in self.layers:
            x = conv(x)
        x_out = self.conv_last(x)
        return x, x_out


class DilatedConvModule(nn.Module):
    def __init__(self, in_ch: int, f_channels: list[int] = (128, 64, 32), out_ch: int = 2):
        super().__init__()
        assert 1 <= len(f_channels) <= 5
        layers = []

        pre_ch = in_ch
        for i, ch in enumerate(f_channels):
            layers.append(ConvNormAct(pre_ch, ch, 3, 1, 2**i))
            pre_ch = ch

        self.layer = nn.Sequential(*layers)
        self.last_conv = nn.Conv2d(pre_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return self.last_conv(x)


class UPFlowNetLite(nn.Module):
    def __init__(
        self,
        encoder_chs: list[int] = [3, 16, 32, 64, 96, 128],
        output_shrink_level: int = 2,  # output size is 1/(1<<output_shrink_level) of the input size
        search_range: int = 2,
        mid_feature_ch: int = 32,
        flow_estimator_conv_chs: list[int] = (64, 32, 16),
        flow_refiner_conv_chs: list[int] = (32, 32, 32, 32, 32),
        normalize_before_cost_volume: bool = True,
        normalize_moments_across_channels: bool = True,
        normalize_moments_across_images: bool = False,  # False recommended for small batch case, though set True in UFlow and UPFlow
        apply_sgu_upsample: bool = True,
    ) -> None:
        super().__init__()
        assert encoder_chs[0] == 3
        assert output_shrink_level >= 0

        self.output_shrink_level = output_shrink_level
        self.apply_sgu_upsample = apply_sgu_upsample
        self.normalize_before_cost_volume = normalize_before_cost_volume
        self.normalize_moments_across_channels = normalize_moments_across_channels
        self.normalize_moments_across_images = normalize_moments_across_images

        dim_corr = (search_range * 2 + 1) ** 2
        flow_estimator_in_ch = dim_corr + mid_feature_ch + 2
        flow_refiner_in_ch = flow_estimator_conv_chs[-1] + 2

        self.encoder = FeatureExtractor(encoder_chs, if_end_relu=True, if_end_norm=False)
        self.midconv = nn.ModuleList(
            [ConvNormAct(ch, mid_feature_ch, kernel_size=1) for ch in encoder_chs]
        )
        self.sgu_upsample = SGUModel(mid_feature_ch)
        self.correlation = Correlation(kernel_size=1, max_displacement=search_range)
        self.residual_flow_estimator = FlowEstimator(
            flow_estimator_in_ch, flow_estimator_conv_chs, out_ch=2
        )
        self.flow_refiner = (
            DilatedConvModule(flow_refiner_in_ch, flow_refiner_conv_chs, out_ch=2)
            if flow_refiner_conv_chs
            else None
        )

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        # encoder
        img1_pyramid = self.encoder(img1)
        img2_pyramid = self.encoder(img2)

        # middle layer
        img1_mid_pyramid = []
        img2_mid_pyramid = []
        for level, (x1, x2) in enumerate(zip(img1_pyramid, img2_pyramid)):
            if level < self.output_shrink_level:
                img1_mid_pyramid.append(None)
                img2_mid_pyramid.append(None)
            else:
                img1_mid_pyramid.append(self.midconv[level](x1))
                img2_mid_pyramid.append(self.midconv[level](x2))

        # decoder
        flows = []
        for level in range(len(img1_pyramid) - 1, self.output_shrink_level - 1, -1):
            x1 = img1_pyramid[level]
            x2 = img2_pyramid[level]
            x1_mid = img1_mid_pyramid[level]
            x2_mid = img2_mid_pyramid[level]

            if level == len(img1_pyramid) - 1:
                # initial flow
                b, _, h0, w0 = x1.shape
                flow = torch.zeros(b, 2, h0, w0, device=img1.device)
                x2_warped = x2
            else:
                # upscale flow
                flow = upsample2d_flow(
                    flow, x1.shape[-2], x1.shape[-1], mode="bilinear", if_rate=True
                )
                if self.apply_sgu_upsample:
                    flow = self.sgu_upsample(flow, x1_mid, x2_mid)
                x2_warped = backwarp(x2, flow)

            # normalize feature
            if self.normalize_before_cost_volume:
                x1, x2_warped = normalize_features(
                    (x1, x2_warped),
                    normalize=True,
                    center=True,
                    moments_across_channels=self.normalize_moments_across_channels,
                    moments_across_images=self.normalize_moments_across_images,
                )

            # correlation
            correlation = self.correlation(x1, x2_warped)
            correlation = F.leaky_relu_(correlation, negative_slope=0.1)
            x1_res, flow_res = self.residual_flow_estimator(
                torch.cat([correlation, x1_mid, flow], dim=1)
            )
            flow = flow + flow_res

            # additional refinement
            if self.flow_refiner:
                flow_fine = self.flow_refiner(torch.cat([x1_res, flow], dim=1))
                flow = flow + flow_fine

            flows.append(flow)

        return flows[::-1]
