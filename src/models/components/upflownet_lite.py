from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.basic_conv import ConvNormAct
from src.models.components.flow_utils import *


class FeatureExtractor(nn.Module):
    def __init__(self, num_chs: list[int], if_end_relu: bool = True, if_end_norm: bool = False):
        super().__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList()
        for ch_in, ch_out in zip(num_chs[:-1], num_chs[1:]):
            self.convs.append(
                nn.Sequential(
                    ConvNormAct(ch_in, ch_out, stride=2),
                    ConvNormAct(ch_out, ch_out,
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


class WarpingLayer_no_div(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), dim=1) + 0.5
        vgrid = grid + flow
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / W - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / H - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
        x_warp = F.grid_sample(x, vgrid, padding_mode="zeros", align_corners=False)
        # mask
        mask = torch.ones(x.size(), device=x.device, requires_grad=False)
        mask = F.grid_sample(mask, vgrid, align_corners=False)
        mask = (mask >= 1.0).float()
        return x_warp * mask




class FlowEstimatorDense_v2(nn.Module):
    def __init__(
        self, ch_in: int, f_channels: list[int] = (128, 128, 96, 64, 32), out_channel: int = 2
    ):
        super().__init__()
        N = 0
        ind = 0
        N += ch_in
        self.conv1 = conv(N, f_channels[ind])
        N += f_channels[ind]

        ind += 1
        self.conv2 = conv(N, f_channels[ind])
        N += f_channels[ind]

        ind += 1
        self.conv3 = conv(N, f_channels[ind])
        N += f_channels[ind]

        ind += 1
        self.conv4 = conv(N, f_channels[ind])
        N += f_channels[ind]

        ind += 1
        self.conv5 = conv(N, f_channels[ind])
        N += f_channels[ind]
        self.n_channels = N
        ind += 1
        self.conv_last = conv(N, out_channel, isReLU=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class ContextNetwork_v2_(nn.Module):
    def __init__(self, ch_in: int, f_channels: list[int] = (128, 128, 128, 96, 64, 32, 2)):
        super().__init__()
        self.convs = nn.Sequential(
            conv(ch_in, f_channels[0], 3, 1, 1),
            conv(f_channels[0], f_channels[1], 3, 1, 2),
            conv(f_channels[1], f_channels[2], 3, 1, 4),
            conv(f_channels[2], f_channels[3], 3, 1, 8),
            conv(f_channels[3], f_channels[4], 3, 1, 16),
            conv(f_channels[4], f_channels[5], 3, 1, 1),
            conv(f_channels[5], f_channels[6], isReLU=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)



class UPFlowNet(nn.Module):
    def __init__(
        self,
        num_chs: list[int] = [3, 16, 32, 64, 96, 128, 196],
        output_level: int = 2,  # decode until num_chs[output_level]
        estimator_f_channels: list[int] = (128, 128, 96, 64, 32),
        context_f_channels: list[int] = (128, 128, 128, 96, 64, 32, 2),
        search_range: int = 4,
        if_norm_before_cost_volume: bool = True,
        norm_moments_across_channels: bool = True,
        norm_moments_across_images: bool = False,  # False recommended for small batch case, though set True in UFlow and UPFlow
        if_sgu_upsample: bool = True,
    ) -> None:
        super().__init__()
        # === build the network
        self.num_chs = num_chs  # [1/2, 1/4, 1/8, 1/16, 1/32, 1/64]
        self.output_level = output_level
        self.search_range = search_range
        self.if_norm_before_cost_volume = if_norm_before_cost_volume
        self.norm_moments_across_channels = norm_moments_across_channels
        self.norm_moments_across_images = norm_moments_across_images
        self.if_sgu_upsample = if_sgu_upsample

        self.estimator_f_channels = estimator_f_channels
        self.context_f_channels = context_f_channels
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = self.dim_corr + 32 + 2

        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs)
        self.warping_layer = WarpingLayer_no_div()
        self.flow_estimators = FlowEstimatorDense_v2(
            self.num_ch_in, f_channels=self.estimator_f_channels
        )
        self.context_networks = ContextNetwork_v2_(
            self.flow_estimators.n_channels + 2, f_channels=self.context_f_channels
        )
        self.conv_1x1 = nn.ModuleList(
            [conv(ch, 32, kernel_size=1, stride=1, dilation=1) for ch in num_chs[::-1]]
        )
        self.correlation_pytorch = Corr_pyTorch(kernel_size=1, max_displacement=self.search_range)
        # === build sgu upsampling
        if self.if_sgu_upsample:
            self.sgi_model = SGUModel()
        else:
            self.sgi_model = None

    def forward(self, x1_raw: torch.Tensor, x2_raw: torch.Tensor):
        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        flows = []
        # init
        b_size, _, h_x1, w_x1 = x1_pyramid[0].shape
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, device=x1_raw.device)
        # build pyramid
        feature_level_ls = []
        for level, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            x1_1by1 = self.conv_1x1[level](x1)
            x2_1by1 = self.conv_1x1[level](x2)
            feature_level_ls.append((x1, x1_1by1, x2, x2_1by1))
            if level == len(x1_pyramid) - 1 - self.output_level:
                break
        for level, (x1, x1_1by1, x2, x2_1by1) in enumerate(feature_level_ls):
            flow_f, flow_f_res = self.decode_level_res(
                level=level,
                flow_1=flow_f,
                feature_1=x1,
                feature_1_1x1=x1_1by1,
                feature_2=x2,
                feature_2_1x1=x2_1by1,
                img_ori_1=x1_raw,
                img_ori_2=x2_raw,
            )
            flow_f = flow_f + flow_f_res
            flows.append(flow_f)
        flow_f_out = upsample2d_flow_as(flow_f, x1_raw, mode="bilinear", if_rate=True)

        # === upsample to full resolution
        if self.if_sgu_upsample:
            feature_1_1x1 = self.sgi_model.output_conv(x1_raw)
            feature_2_1x1 = self.sgi_model.output_conv(x2_raw)
            flow_f_out = self.self_guided_upsample(
                flow_up_bilinear=flow_f,
                feature_1=feature_1_1x1,
                feature_2=feature_2_1x1,
                output_level_flow=flow_f_out,
            )

        flows.append(flow_f_out)

        return flows[::-1]

    def decode_level_res(
        self,
        level: int,
        flow_1: torch.Tensor,
        feature_1: torch.Tensor,
        feature_1_1x1: torch.Tensor,
        feature_2: torch.Tensor,
        feature_2_1x1: torch.Tensor,
        img_ori_1: torch.Tensor,
        img_ori_2: torch.Tensor,
    ):
        flow_1_up_bilinear = upsample2d_flow_as(flow_1, feature_1, mode="bilinear", if_rate=True)

        # warping
        if level == 0:
            feature_2_warp = feature_2
        else:
            if self.if_sgu_upsample:
                flow_1_up_bilinear = self.self_guided_upsample(
                    flow_up_bilinear=flow_1_up_bilinear,
                    feature_1=feature_1_1x1,
                    feature_2=feature_2_1x1,
                )
            feature_2_warp = self.warping_layer(feature_2, flow_1_up_bilinear)

        # if norm feature
        if self.if_norm_before_cost_volume:
            feature_1, feature_2_warp = normalize_features(
                (feature_1, feature_2_warp),
                normalize=True,
                center=True,
                moments_across_channels=self.norm_moments_across_channels,
                moments_across_images=self.norm_moments_across_images,
            )

        # correlation
        out_corr_1 = self.correlation_pytorch(feature_1, feature_2_warp)
        out_corr_relu_1 = F.leaky_relu_(out_corr_1, negative_slope=0.1)

        feature_int_1, flow_res_1 = self.flow_estimators(
            torch.cat([out_corr_relu_1, feature_1_1x1, flow_1_up_bilinear], dim=1)
        )
        flow_1_up_bilinear_ = flow_1_up_bilinear + flow_res_1
        flow_fine_1 = self.context_networks(torch.cat([feature_int_1, flow_1_up_bilinear_], dim=1))
        flow_1_res = flow_res_1 + flow_fine_1

        return flow_1_up_bilinear, flow_1_res

    def self_guided_upsample(self, flow_up_bilinear, feature_1, feature_2, output_level_flow=None):
        flow_up_bilinear_, out_flow, inter_flow, inter_mask = self.sgi_model(
            flow_up_bilinear, feature_1, feature_2, output_level_flow=output_level_flow
        )
        return out_flow


################################################################


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
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=kernel_size//2, stride=1)
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
    def __init__(
        self,
        in_ch: int,
        f_channels: list[int] = (128, 64, 32),
        out_ch: int = 2
    ):
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
        self.midconv = nn.ModuleList([ConvNormAct(ch, mid_feature_ch, kernel_size=1) for ch in encoder_chs])
        self.sgu_upsample = SGUModel(mid_feature_ch)
        self.correlation = Correlation(kernel_size=1, max_displacement=search_range)
        self.residual_flow_estimator = FlowEstimator(flow_estimator_in_ch, flow_estimator_conv_chs, out_ch=2)
        self.flow_refiner = DilatedConvModule(flow_refiner_in_ch, flow_refiner_conv_chs, out_ch=2) \
            if flow_refiner_conv_chs else None
            
    
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
        for level in range(len(img1_pyramid)-1, self.output_shrink_level-1, -1):
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
                flow = upsample2d_flow(flow, x1.shape[-2], x1.shape[-1], mode="bilinear", if_rate=True)
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
