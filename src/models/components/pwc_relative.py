from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def custom_grid_sample(tensor: torch.Tensor, res_grid: torch.Tensor) -> torch.Tensor:
        """ grid_sample for relative residual grid flow"""
        xs = torch.linspace(0.5, tensor.shape[-1]-0.5, steps=tensor.shape[-1]) / tensor.shape[-1] * 2 - 1
        ys = torch.linspace(0.5, tensor.shape[-2]-0.5, steps=tensor.shape[-2]) / tensor.shape[-2] * 2 - 1
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        base_grid = torch.stack([x, y], dim=-1)
        return F.grid_sample(x, base_grid + res_grid)


class FeatureExtractor(nn.Module):
    def __init__(self, dims: list[int] = [16, 32, 64, 96]) -> None:
        super().__init__()
        assert len(dims), f"{dims=} must have at least one element"
        assert min(dims) > 0, f"{dims=} must be all positive"
        assert sorted(dims) == dims, f"{dims=} must be increasing order"
        self.layers = []

        pd = 3
        for d in dims:
            self.layers.append(nn.Sequential(
                nn.Conv2d(pd, d, kernel_size=3, stride=2, padding=1),
                nn.PReLU(),
                nn.Conv2d(d, d, kernel_size=3, padding=1),
                nn.PReLU(),
                nn.Conv2d(d, d, kernel_size=3, padding=1),
                nn.PReLU(),
            ))
            pd = d
        
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        ret = []
        for layer in self.layers:
            x = layer(x)
            ret.append(x)
        return ret


class CostVolume(nn.Module):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    def __init__(self, search_range: int = 2) -> None:
        super().__init__()
        self.search_range = search_range
        self.max_offset = search_range * 2 + 1
        self.prelu = nn.PReLU()

    def forward(self, c1: torch.Tensor, c2_backwarped: torch.Tensor) -> torch.Tensor:
        padded_lvl = F.pad(c2_backwarped, [[0, 0], [0, 0], [self.search_range, self.search_range], [self.search_range, self.search_range]], mode='reflect')
        h, w = c1.shape[-2:]

        cost_vol = []
        for y in range(self.max_offset):
            for x in range(self.max_offset):
                slice = padded_lvl[:,:,y:y+h, x:x+w]
                cost = torch.mean(c1 * slice, dim=1, keepdim=True)
                cost_vol.append(cost)
        cost_vol = torch.cat(cost_vol, axis=1)

        return self.prelu(cost_vol)


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, coarse_feat_dim: bool, search_range: int = 2) -> None:
        super().__init__()
        assert coarse_feat_dim >= 0
        self.lowest_stage = coarse_feat_dim == 0
        self.costVolume = CostVolume(search_range)

        concat_dim = dim * (search_range * search_range * 4 + 1) + coarse_feat_dim + (2 if coarse_feat_dim > 0 else 0)
        self.layer = nn.Sequential(
            nn.Conv2d(concat_dim, dim, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
        )
        self.predictor = nn.Conv2d(dim, 2, kernel_size=3, padding=1)
    
    def forward(self, c1: torch.Tensor, c2: torch.Tensor, coarse_flow: Optional[torch.Tensor], coarse_feat: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.lowest_stage:
            c2_backwarped = custom_grid_sample(c2, coarse_flow)
            corr = self.costVolume(c1, c2_backwarped)
            refined_feat = torch.cat([corr, c1, coarse_flow, coarse_feat], dim=1)
        else:
            corr = self.costVolume(c1, c2)
            refined_feat = torch.cat([corr, c1], dim=1)
        
        refined_feat = self.layer(refined_feat)
        refined_flow = self.predictor(refined_feat)

        return refined_flow, refined_feat


class PWCNet(nn.Module):
    def __init__(self, dims: list[int] = [16, 32, 64, 96], search_range: int = 2) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(dims)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d, pd, search_range) for d, pd in zip(dims, dims[1:] + [0])
        ])
        self.upfeats = nn.ModuleList([
            nn.ConvTranspose2d(d, 2, kernel_size=4, stride=2) for d in dims
        ])
        self.upflows = nn.ModuleList([
            nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2) for _ in dims
        ])
        self.last_feat = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, padding=1),
            nn.PReLU(),
        )
        self.last_flow = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(dims[0], 2, kernel_size=3, padding=1),
        )
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        c1_list = self.feature_extractor(img1)
        c2_list = self.feature_extractor(img2)

        flows = []
        coarse_flow, coarse_feat = None, None
        for i in range(len(c1_list)-1, -1, -1):
            c1, c2 = c1_list[i], c2_list[i]
            decoder = self.decoder_blocks[i]
            coarse_flow, coarse_feat = decoder(c1, c2, coarse_flow, coarse_feat)
            flows.append(coarse_flow)
            coarse_flow = self.upflows[i](coarse_flow)
            coarse_feat = self.upfeats[i](coarse_feat)
        
        coarse_feat = self.last_feat(coarse_feat)
        coarse_flow += self.last_flow(coarse_feat)
        flows.append(coarse_flow)

        return flows[::-1]
