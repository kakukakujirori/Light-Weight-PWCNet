import torch
import torch.nn as nn


class EPE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        norm = torch.sum((pred - gt)**2, dim=1)**0.5
        return norm.mean()