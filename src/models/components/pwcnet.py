"""Mainly borrowed from https://github.com/hellochick/PWCNet-tf2 There should be a room to improve
efficiency for computing costvolume."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def bilinear_warp(x: torch.Tensor, flow: torch.Tensor):
    h, w = x.shape[-2:]
    grid_x, grid_y = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="xy")
    grid_x = grid_x.reshape(1, 1, *grid_y.shape)
    grid_y = grid_y.reshape(1, 1, *grid_y.shape)

    fx, fy = torch.chunk(flow, chunks=2, dim=1)
    gx, gy = grid_x + fx, grid_y + fy

    # normalize
    gx = (gx + 0.5) * 2 / w - 1
    gy = (gy + 0.5) * 2 / h - 1
    gxy = torch.cat([gx, gy], dim=1).permute(0, 2, 3, 1)

    return F.grid_sample(x, gxy)


class ConvPReLU(nn.Module):
    def __init__(
        self, ch_in: int, ch_out: int, stride: int = 1, padding: int = 1, dilation: int = 1
    ):
        super().__init__()
        self.conv_out = nn.Sequential(
            nn.Conv2d(
                ch_in, ch_out, kernel_size=3, stride=stride, padding=padding, dilation=dilation
            ),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(x)


class DeConv(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, scale: float = 1):
        super().__init__()
        self.deconv_out = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv_out(x) * self.scale


class CostVolumn(nn.Module):
    def __init__(self, search_range: int):
        super().__init__()
        self.search_range = search_range
        self.max_offset = search_range * 2 + 1
        self.relu = nn.PReLU()

    def forward(self, c1: torch.Tensor, warped: torch.Tensor):
        """Build cost volume for associating a pixel from Image1 with its corresponding pixels in
        Image2.

        Args:
            c1: Level of the feature pyramid of Image1
            warp: Warped level of the feature pyramid of image22
            search_range: Search range (maximum displacement)
        """
        padded_lvl = F.pad(
            warped, (self.search_range, self.search_range, self.search_range, self.search_range)
        )
        h, w = c1.shape[-2:]

        cost_vol = []
        for y in range(self.max_offset):
            for x in range(self.max_offset):
                slice = padded_lvl[:, :, y : (y + h), x : (x + w)]
                cost = torch.mean(c1 * slice, axis=1, keepdims=True)
                cost_vol.append(cost)
        cost_vol = torch.cat(cost_vol, axis=1)
        cost_vol = self.relu(cost_vol)

        return cost_vol  # [B, self.max_offset**2, H, W]


class PWCNet(nn.Module):
    def __init__(self, search_range: int = 4):
        super().__init__()
        self.conv1a = ConvPReLU(3, 16, stride=2)
        self.conv1aa = ConvPReLU(16, 16, stride=1)
        self.conv1b = ConvPReLU(16, 16, stride=1)
        self.conv2a = ConvPReLU(16, 32, stride=2)
        self.conv2aa = ConvPReLU(32, 32, stride=1)
        self.conv2b = ConvPReLU(32, 32, stride=1)
        self.conv3a = ConvPReLU(32, 64, stride=2)
        self.conv3aa = ConvPReLU(64, 64, stride=1)
        self.conv3b = ConvPReLU(64, 64, stride=1)
        self.conv4a = ConvPReLU(64, 96, stride=2)
        self.conv4aa = ConvPReLU(96, 96, stride=1)
        self.conv4b = ConvPReLU(96, 96, stride=1)
        self.conv5a = ConvPReLU(96, 128, stride=2)
        self.conv5aa = ConvPReLU(128, 128, stride=1)
        self.conv5b = ConvPReLU(128, 128, stride=1)
        self.conv6a = ConvPReLU(128, 196, stride=2)
        self.conv6aa = ConvPReLU(196, 196, stride=1)
        self.conv6b = ConvPReLU(196, 196, stride=1)

        costvol_ch = (2 * search_range + 1) ** 2

        self.costv_6 = CostVolumn(search_range)
        self.conv6_0 = ConvPReLU(costvol_ch, 128, stride=1)
        self.conv6_1 = ConvPReLU(costvol_ch + 128, 128, stride=1)
        self.conv6_2 = ConvPReLU(costvol_ch + 128 + 128, 96, stride=1)
        self.conv6_3 = ConvPReLU(costvol_ch + 128 + 128 + 96, 64, stride=1)
        self.conv6_4 = ConvPReLU(costvol_ch + 128 + 128 + 96 + 64, 32, stride=1)
        self.deconv6 = DeConv(2, 2, scale=2)
        self.upfeat6 = DeConv(costvol_ch + 128 + 128 + 96 + 64 + 32, 2)

        self.costv_5 = CostVolumn(search_range)
        self.conv5_0 = ConvPReLU(costvol_ch + 128 + 2 + 2, 128, stride=1)
        self.conv5_1 = ConvPReLU(costvol_ch + 128 + 2 + 2 + 128, 128, stride=1)
        self.conv5_2 = ConvPReLU(costvol_ch + 128 + 2 + 2 + 128 + 128, 96, stride=1)
        self.conv5_3 = ConvPReLU(costvol_ch + 128 + 2 + 2 + 128 + 128 + 96, 64, stride=1)
        self.conv5_4 = ConvPReLU(costvol_ch + 128 + 2 + 2 + 128 + 128 + 96 + 64, 32, stride=1)
        self.deconv5 = DeConv(2, 2, scale=2)
        self.upfeat5 = DeConv(costvol_ch + 128 + 2 + 2 + 128 + 128 + 96 + 64 + 32, 2)

        self.costv_4 = CostVolumn(search_range)
        self.conv4_0 = ConvPReLU(costvol_ch + 96 + 2 + 2, 128, stride=1)
        self.conv4_1 = ConvPReLU(costvol_ch + 96 + 2 + 2 + 128, 128, stride=1)
        self.conv4_2 = ConvPReLU(costvol_ch + 96 + 2 + 2 + 128 + 128, 96, stride=1)
        self.conv4_3 = ConvPReLU(costvol_ch + 96 + 2 + 2 + 128 + 128 + 96, 64, stride=1)
        self.conv4_4 = ConvPReLU(costvol_ch + 96 + 2 + 2 + 128 + 128 + 96 + 64, 32, stride=1)
        self.deconv4 = DeConv(2, 2, scale=2)
        self.upfeat4 = DeConv(costvol_ch + 96 + 2 + 2 + 128 + 128 + 96 + 64 + 32, 2)

        self.costv_3 = CostVolumn(search_range)
        self.conv3_0 = ConvPReLU(costvol_ch + 64 + 2 + 2, 128, stride=1)
        self.conv3_1 = ConvPReLU(costvol_ch + 64 + 2 + 2 + 128, 128, stride=1)
        self.conv3_2 = ConvPReLU(costvol_ch + 64 + 2 + 2 + 128 + 128, 96, stride=1)
        self.conv3_3 = ConvPReLU(costvol_ch + 64 + 2 + 2 + 128 + 128 + 96, 64, stride=1)
        self.conv3_4 = ConvPReLU(costvol_ch + 64 + 2 + 2 + 128 + 128 + 96 + 64, 32, stride=1)
        self.deconv3 = DeConv(2, 2, scale=2)
        self.upfeat3 = DeConv(costvol_ch + 64 + 2 + 2 + 128 + 128 + 96 + 64 + 32, 2)

        self.costv_2 = CostVolumn(search_range)
        self.conv2_0 = ConvPReLU(costvol_ch + 32 + 2 + 2, 128, stride=1)
        self.conv2_1 = ConvPReLU(costvol_ch + 32 + 2 + 2 + 128, 128, stride=1)
        self.conv2_2 = ConvPReLU(costvol_ch + 32 + 2 + 2 + 128 + 128, 96, stride=1)
        self.conv2_3 = ConvPReLU(costvol_ch + 32 + 2 + 2 + 128 + 128 + 96, 64, stride=1)
        self.conv2_4 = ConvPReLU(costvol_ch + 32 + 2 + 2 + 128 + 128 + 96 + 64, 32, stride=1)
        self.deconv2 = DeConv(2, 2, scale=2)

        self.dc_conv1 = ConvPReLU(
            costvol_ch + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32,
            128,
            stride=1,
            padding=1,
            dilation=1,
        )
        self.dc_conv2 = ConvPReLU(128, 128, stride=1, padding=2, dilation=2)
        self.dc_conv3 = ConvPReLU(128, 128, stride=1, padding=4, dilation=4)
        self.dc_conv4 = ConvPReLU(128, 96, stride=1, padding=8, dilation=8)
        self.dc_conv5 = ConvPReLU(96, 64, stride=1, padding=16, dilation=16)
        self.dc_conv6 = ConvPReLU(64, 32, stride=1, padding=1, dilation=1)
        self.dc_conv7 = nn.Conv2d(32, 2, kernel_size=3, padding=1)

        self.predict_flow6 = nn.Conv2d(
            costvol_ch + 128 + 128 + 96 + 64 + 32, 2, kernel_size=3, padding=1
        )
        self.predict_flow5 = nn.Conv2d(
            costvol_ch + 128 + 2 + 2 + 128 + 128 + 96 + 64 + 32, 2, kernel_size=3, padding=1
        )
        self.predict_flow4 = nn.Conv2d(
            costvol_ch + 96 + 2 + 2 + 128 + 128 + 96 + 64 + 32, 2, kernel_size=3, padding=1
        )
        self.predict_flow3 = nn.Conv2d(
            costvol_ch + 64 + 2 + 2 + 128 + 128 + 96 + 64 + 32, 2, kernel_size=3, padding=1
        )
        self.predict_flow2 = nn.Conv2d(
            costvol_ch + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, 2, kernel_size=3, padding=1
        )

    def forward(self, im1: torch.Tensor, im2: torch.Tensor, is_training=True):
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6aa(self.conv6a(c15)))
        c26 = self.conv6b(self.conv6aa(self.conv6a(c25)))

        # 6th flow
        corr6 = self.costv_6(c1=c16, warped=c26)
        x = torch.cat([self.conv6_0(corr6), corr6], dim=1)
        x = torch.cat([self.conv6_1(x), x], dim=1)
        x = torch.cat([self.conv6_2(x), x], dim=1)
        x = torch.cat([self.conv6_3(x), x], dim=1)
        x = torch.cat([self.conv6_4(x), x], dim=1)

        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        # 5th flow
        warp5 = bilinear_warp(c25, up_flow6)
        corr5 = self.costv_5(c1=c15, warped=warp5)

        x = torch.cat([corr5, c15, up_flow6, up_feat6], 1)
        x = torch.cat([self.conv5_0(x), x], 1)
        x = torch.cat([self.conv5_1(x), x], 1)
        x = torch.cat([self.conv5_2(x), x], 1)
        x = torch.cat([self.conv5_3(x), x], 1)
        x = torch.cat([self.conv5_4(x), x], 1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        # 4th flow
        warp4 = bilinear_warp(c24, up_flow5)
        corr4 = self.costv_4(c1=c14, warped=warp4)

        x = torch.cat([corr4, c14, up_flow5, up_feat5], 1)
        x = torch.cat([self.conv4_0(x), x], 1)
        x = torch.cat([self.conv4_1(x), x], 1)
        x = torch.cat([self.conv4_2(x), x], 1)
        x = torch.cat([self.conv4_3(x), x], 1)
        x = torch.cat([self.conv4_4(x), x], 1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        # 3rd flow
        warp3 = bilinear_warp(c23, up_flow4)
        corr3 = self.costv_3(c1=c13, warped=warp3)

        x = torch.cat([corr3, c13, up_flow4, up_feat4], 1)
        x = torch.cat([self.conv3_0(x), x], 1)
        x = torch.cat([self.conv3_1(x), x], 1)
        x = torch.cat([self.conv3_2(x), x], 1)
        x = torch.cat([self.conv3_3(x), x], 1)
        x = torch.cat([self.conv3_4(x), x], 1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        # 2nd flow
        warp2 = bilinear_warp(c22, up_flow3)
        corr2 = self.costv_2(c1=c12, warped=warp2)

        x = torch.cat([corr2, c12, up_flow3, up_feat3], dim=1)
        x = torch.cat([self.conv2_0(x), x], dim=1)
        x = torch.cat([self.conv2_1(x), x], dim=1)
        x = torch.cat([self.conv2_2(x), x], dim=1)
        x = torch.cat([self.conv2_3(x), x], dim=1)
        x = torch.cat([self.conv2_4(x), x], dim=1)
        flow2 = self.predict_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        return flow2, flow3, flow4, flow5, flow6
