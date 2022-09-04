"""Borrowed from https://github.com/coolbeam/UPFlow_pytorch."""
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    dilation: int = 1,
    isReLU: bool = True,
    if_IN: bool = False,
    IN_affine: bool = False,
    if_BN: bool = False,
):
    if isReLU:
        if if_IN:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=((kernel_size - 1) * dilation) // 2,
                    bias=True,
                ),
                nn.LeakyReLU(0.1, inplace=True),
                nn.InstanceNorm2d(out_planes, affine=IN_affine),
            )
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=((kernel_size - 1) * dilation) // 2,
                    bias=True,
                ),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(out_planes, affine=IN_affine),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=((kernel_size - 1) * dilation) // 2,
                    bias=True,
                ),
                nn.LeakyReLU(0.1, inplace=True),
            )
    else:
        if if_IN:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=((kernel_size - 1) * dilation) // 2,
                    bias=True,
                ),
                nn.InstanceNorm2d(out_planes, affine=IN_affine),
            )
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=((kernel_size - 1) * dilation) // 2,
                    bias=True,
                ),
                nn.BatchNorm2d(out_planes, affine=IN_affine),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=((kernel_size - 1) * dilation) // 2,
                    bias=True,
                )
            )


def upsample2d_flow_as(
    inputs: torch.Tensor, target_as: torch.Tensor, mode: str = "bilinear", if_rate: bool = True
) -> torch.Tensor:
    _, _, h, w = target_as.size()
    res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
    if if_rate:
        _, _, h_, w_ = inputs.size()
        res[:, 0, :, :] *= w / w_
        res[:, 1, :, :] *= h / h_
    return res


def normalize_features(
    feature_list: list[torch.Tensor],
    normalize: bool,
    center: bool,
    moments_across_channels: bool = True,
    moments_across_images: bool = True,
) -> list[torch.Tensor]:
    """Normalizes feature tensors (e.g., before computing the cost volume).

    Args:
      feature_list: list of torch tensors, each with dimensions [b, c, h, w]
      normalize: bool flag, divide features by their standard deviation
      center: bool flag, subtract feature mean
      moments_across_channels: bool flag, compute mean and std across channels, 看到UFlow默认是True
      moments_across_images: bool flag, compute mean and std across images, 看到UFlow默认是True
    Returns:
      list, normalized feature_list
    """
    # Compute feature statistics.
    statistics = defaultdict(list)
    axes = [1, 2, 3] if moments_across_channels else [2, 3]  # [b, c, h, w]
    for feature_image in feature_list:
        mean = torch.mean(feature_image, dim=axes, keepdim=True)  # [b,1,1,1] or [b,c,1,1]
        variance = torch.var(feature_image, dim=axes, keepdim=True)  # [b,1,1,1] or [b,c,1,1]
        statistics["mean"].append(mean)
        statistics["var"].append(variance)

    if moments_across_images:
        statistics["mean"] = [torch.mean(torch.stack(statistics["mean"], dim=0), dim=(0,))] * len(
            feature_list
        )
        statistics["var"] = [torch.var(torch.stack(statistics["var"], dim=0), dim=(0,))] * len(
            feature_list
        )

    statistics["std"] = [torch.sqrt(v + 1e-16) for v in statistics["var"]]

    # Center and normalize features.
    if center:
        feature_list = [f - mean for f, mean in zip(feature_list, statistics["mean"])]
    if normalize:
        feature_list = [f / std for f, std in zip(feature_list, statistics["std"])]

    return feature_list


def torch_warp(x: torch.Tensor, flo: torch.Tensor) -> torch.Tensor:
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, _, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1) + 0.5
    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / W - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / H - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
    output = F.grid_sample(x, vgrid, padding_mode="zeros", align_corners=False)
    return output


class FeatureExtractor(nn.Module):
    def __init__(self, num_chs: list[int], if_end_relu: bool = True, if_end_norm: bool = False):
        super().__init__()
        self.num_chs = num_chs
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    conv(ch_in, ch_out, stride=2),
                    conv(ch_out, ch_out, isReLU=if_end_relu, if_IN=if_end_norm),
                )
                for ch_in, ch_out in zip(num_chs[:-1], num_chs[1:])
            ]
        )

    def forward(self, x: torch.Tensor):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


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


class Corr_pyTorch(nn.Module):
    """my implementation of correlation layer using pytorch note that the Ispeed is much slower
    than cuda version."""

    def __init__(self, kernel_size: int = 1, max_displacement: int = 4):
        super().__init__()
        self.pad_size = max_displacement
        self.kernel_size = kernel_size
        self.stride1 = 1
        self.stride2 = 1
        self.padlayer = nn.ConstantPad2d(self.pad_size, 0)

    def forward(self, in1: torch.Tensor, in2: torch.Tensor) -> torch.Tensor:
        bz, _, hei, wid = in1.shape
        f1 = F.unfold(
            in1, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=self.stride1
        )
        f2 = F.unfold(
            in2, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=self.stride2
        )
        searching_kernel_size = f2.shape[1]
        f2_ = torch.reshape(f2, (bz, searching_kernel_size, hei, wid))
        f2_ = torch.reshape(f2_, (bz * searching_kernel_size, hei, wid)).unsqueeze(1)
        f2 = F.unfold(f2_, kernel_size=(hei, wid), padding=self.pad_size, stride=self.stride2)
        _, kernel_number, window_number = f2.shape
        f2_ = torch.reshape(f2, (bz, searching_kernel_size, kernel_number, window_number))
        f2_2 = torch.transpose(f2_, dim0=1, dim1=3).transpose(2, 3)
        f1_2 = f1.unsqueeze(1)
        res = f2_2 * f1_2
        res = torch.mean(res, dim=2)
        res = torch.reshape(res, (bz, window_number, hei, wid))
        return res


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


class FlowEstimatorDense_temp(nn.Module):
    def __init__(
        self, ch_in: int, f_channels: list[int] = (128, 128, 96, 64, 32), ch_out: int = 2
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
        self.num_feature_channel = N
        ind += 1
        self.conv_last = conv(N, ch_out, isReLU=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        return x5, x_out


class SGUModel(nn.Module):
    def __init__(self):
        super().__init__()
        f_channels_es = (32, 32, 32, 16, 8)
        in_C = 64
        self.warping_layer = WarpingLayer_no_div()
        self.dense_estimator_mask = FlowEstimatorDense_temp(
            in_C, f_channels=f_channels_es, ch_out=3
        )
        self.upsample_output_conv = nn.Sequential(
            conv(3, 16, kernel_size=3, stride=1, dilation=1),
            conv(16, 16, stride=2),
            conv(16, 32, kernel_size=3, stride=1, dilation=1),
            conv(32, 32, stride=2),
        )

    def forward(self, flow_init, feature_1, feature_2, output_level_flow=None):
        n, c, h, w = flow_init.shape
        n_f, c_f, h_f, w_f = feature_1.shape
        if h != h_f or w != w_f:
            flow_init = upsample2d_flow_as(flow_init, feature_1, mode="bilinear", if_rate=True)
        feature_2_warp = self.warping_layer(feature_2, flow_init)
        input_feature = torch.cat((feature_1, feature_2_warp), dim=1)
        feature, x_out = self.dense_estimator_mask(input_feature)
        inter_flow = x_out[:, 0:2, :, :]
        inter_mask = x_out[:, 2:3, :, :]
        inter_mask = torch.sigmoid(inter_mask)
        if output_level_flow is not None:
            inter_flow = upsample2d_flow_as(
                inter_flow, output_level_flow, mode="bilinear", if_rate=True
            )
            inter_mask = upsample2d_flow_as(
                inter_mask, output_level_flow, mode="bilinear", if_rate=False
            )
            flow_init = output_level_flow
        flow_up = torch_warp(flow_init, inter_flow) * (1 - inter_mask) + flow_init * inter_mask
        return flow_init, flow_up, inter_flow, inter_mask

    def output_conv(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample_output_conv(x)


class UPFlowNet(nn.Module):
    def __init__(
        self,
        num_chs: list[int] = [3, 16, 32, 64, 96, 128, 196],
        output_level: int = 4,  # decode until num_chs[:-output_level-1]
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

        self.estimator_f_channels = (128, 128, 96, 64, 32)
        self.context_f_channels = (128, 128, 128, 96, 64, 32, 2)
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
            [
                conv(196, 32, kernel_size=1, stride=1, dilation=1),
                conv(128, 32, kernel_size=1, stride=1, dilation=1),
                conv(96, 32, kernel_size=1, stride=1, dilation=1),
                conv(64, 32, kernel_size=1, stride=1, dilation=1),
                conv(32, 32, kernel_size=1, stride=1, dilation=1),
            ]
        )
        self.correlation_pytorch = Corr_pyTorch(kernel_size=1, max_displacement=self.search_range)
        # === build sgu upsampling
        if self.if_sgu_upsample:
            self.sgi_model = SGUModel()
        else:
            self.sgi_model = None

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        flow_f_pwc_out, flow_b_pwc_out, flows = self.forward_2_frame_v3(
            img1, img2
        )  # forward estimation
        return flow_f_pwc_out, flow_b_pwc_out

    def forward_2_frame_v3(self, x1_raw: torch.Tensor, x2_raw: torch.Tensor):
        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        flows = []
        # init
        (
            b_size,
            _,
            h_x1,
            w_x1,
        ) = x1_pyramid[0].shape
        flow_f = torch.zeros(b_size, 2, h_x1, w_x1, device=x1_raw.device)
        flow_b = torch.zeros(b_size, 2, h_x1, w_x1, device=x1_raw.device)
        # build pyramid
        feature_level_ls = []
        for level, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            x1_1by1 = self.conv_1x1[level](x1)
            x2_1by1 = self.conv_1x1[level](x2)
            feature_level_ls.append((x1, x1_1by1, x2, x2_1by1))
            if level == self.output_level:
                break
        for level, (x1, x1_1by1, x2, x2_1by1) in enumerate(feature_level_ls):
            flow_f, flow_b, flow_f_res, flow_b_res = self.decode_level_res(
                level=level,
                flow_1=flow_f,
                flow_2=flow_b,
                feature_1=x1,
                feature_1_1x1=x1_1by1,
                feature_2=x2,
                feature_2_1x1=x2_1by1,
                img_ori_1=x1_raw,
                img_ori_2=x2_raw,
            )
            flow_f = flow_f + flow_f_res
            flow_b = flow_b + flow_b_res
            flows.append([flow_f, flow_b])
        flow_f_out = upsample2d_flow_as(flow_f, x1_raw, mode="bilinear", if_rate=True)
        flow_b_out = upsample2d_flow_as(flow_b, x1_raw, mode="bilinear", if_rate=True)

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
            flow_b_out = self.self_guided_upsample(
                flow_up_bilinear=flow_b,
                feature_1=feature_2_1x1,
                feature_2=feature_1_1x1,
                output_level_flow=flow_b_out,
            )

        return flow_f_out, flow_b_out, flows[::-1]

    def decode_level_res(
        self,
        level: int,
        flow_1: torch.Tensor,
        flow_2: torch.Tensor,
        feature_1: torch.Tensor,
        feature_1_1x1: torch.Tensor,
        feature_2: torch.Tensor,
        feature_2_1x1: torch.Tensor,
        img_ori_1: torch.Tensor,
        img_ori_2: torch.Tensor,
    ):
        flow_1_up_bilinear = upsample2d_flow_as(flow_1, feature_1, mode="bilinear", if_rate=True)
        flow_2_up_bilinear = upsample2d_flow_as(flow_2, feature_2, mode="bilinear", if_rate=True)
        # warping
        if level == 0:
            feature_2_warp = feature_2
            feature_1_warp = feature_1
        else:
            if self.if_sgu_upsample:
                flow_1_up_bilinear = self.self_guided_upsample(
                    flow_up_bilinear=flow_1_up_bilinear,
                    feature_1=feature_1_1x1,
                    feature_2=feature_2_1x1,
                )
                flow_2_up_bilinear = self.self_guided_upsample(
                    flow_up_bilinear=flow_2_up_bilinear,
                    feature_1=feature_2_1x1,
                    feature_2=feature_1_1x1,
                )
            feature_2_warp = self.warping_layer(feature_2, flow_1_up_bilinear)
            feature_1_warp = self.warping_layer(feature_1, flow_2_up_bilinear)
        # if norm feature
        if self.if_norm_before_cost_volume:
            feature_1, feature_2_warp = normalize_features(
                (feature_1, feature_2_warp),
                normalize=True,
                center=True,
                moments_across_channels=self.norm_moments_across_channels,
                moments_across_images=self.norm_moments_across_images,
            )
            feature_2, feature_1_warp = normalize_features(
                (feature_2, feature_1_warp),
                normalize=True,
                center=True,
                moments_across_channels=self.norm_moments_across_channels,
                moments_across_images=self.norm_moments_across_images,
            )
        # correlation
        out_corr_1 = self.correlation_pytorch(feature_1, feature_2_warp)
        out_corr_2 = self.correlation_pytorch(feature_2, feature_1_warp)
        out_corr_relu_1 = F.leaky_relu_(out_corr_1, negative_slope=0.1)
        out_corr_relu_2 = F.leaky_relu_(out_corr_2, negative_slope=0.1)

        feature_int_1, flow_res_1 = self.flow_estimators(
            torch.cat([out_corr_relu_1, feature_1_1x1, flow_1_up_bilinear], dim=1)
        )
        feature_int_2, flow_res_2 = self.flow_estimators(
            torch.cat([out_corr_relu_2, feature_2_1x1, flow_2_up_bilinear], dim=1)
        )
        flow_1_up_bilinear_ = flow_1_up_bilinear + flow_res_1
        flow_2_up_bilinear_ = flow_2_up_bilinear + flow_res_2
        flow_fine_1 = self.context_networks(torch.cat([feature_int_1, flow_1_up_bilinear_], dim=1))
        flow_fine_2 = self.context_networks(torch.cat([feature_int_2, flow_2_up_bilinear_], dim=1))
        flow_1_res = flow_res_1 + flow_fine_1
        flow_2_res = flow_res_2 + flow_fine_2

        return flow_1_up_bilinear, flow_2_up_bilinear, flow_1_res, flow_2_res

    def self_guided_upsample(self, flow_up_bilinear, feature_1, feature_2, output_level_flow=None):
        flow_up_bilinear_, out_flow, inter_flow, inter_mask = self.sgi_model(
            flow_up_bilinear, feature_1, feature_2, output_level_flow=output_level_flow
        )
        return out_flow
