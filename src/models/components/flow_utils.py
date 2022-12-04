import torch
import torch.nn.functional as F


@torch.jit.script
def upsample2d_flow(
    inputs: torch.Tensor,
    target_height: int,
    target_width: int,
    mode: str = "bilinear",
    if_rate: bool = True,
) -> torch.Tensor:
    res = F.interpolate(inputs, [target_height, target_width], mode=mode, align_corners=False)
    if if_rate:
        _, _, h_, w_ = inputs.size()
        res[:, 0, :, :] *= target_width / w_
        res[:, 1, :, :] *= target_height / h_
    return res


@torch.jit.script
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
    means = []
    vars = []

    axes = [1, 2, 3] if moments_across_channels else [2, 3]  # [b, c, h, w]
    for feature_image in feature_list:
        means.append(torch.mean(feature_image, dim=axes, keepdim=True))  # [b,1,1,1] or [b,c,1,1]
        vars.append(torch.var(feature_image, dim=axes, keepdim=True))  # [b,1,1,1] or [b,c,1,1]

    if moments_across_images:
        means = [torch.mean(torch.stack(means, dim=0), dim=0)] * len(feature_list)
        vars = [torch.var(torch.stack(vars, dim=0), dim=0)] * len(feature_list)

    stds = [torch.sqrt(v + 1e-16) for v in vars]

    # Center and normalize features.
    if center:
        feature_list = [f - mean for f, mean in zip(feature_list, means)]
    if normalize:
        feature_list = [f / std for f, std in zip(feature_list, stds)]

    return feature_list


@torch.jit.script
def backwarp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flow: [B, 2, H, W]
    """
    _, _, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, H, W, 1)
    yy = yy.view(1, H, W, 1)
    grid = torch.cat((xx, yy), -1) + 0.5
    vgrid = grid + flow.permute(0, 2, 3, 1)
    # scale grid to [-1,1]
    vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / W - 1.0
    vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / H - 1.0
    output = F.grid_sample(x, vgrid, padding_mode="zeros", align_corners=False)
    return output
