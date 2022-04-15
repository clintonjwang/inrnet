import torch
nn = torch.nn
F = nn.functional
from functools import partial

class AvgPool(nn.Module):
    def __init__(self, radius, p_norm=2):
        super().__init__()
        self.radius = radius
        self.norm = partial(torch.linalg.norm, ord=p_norm, dim=-1)
    def forward(self, inr):
        return inr.pool(self, "avg")

class MaxPool(nn.Module):
    def __init__(self, radius, p_norm=2):
        super().__init__()
        self.radius = radius
        self.norm = partial(torch.linalg.norm, ord=p_norm, dim=-1)
    def forward(self, inr):
        return inr.pool(self, "max")

class GlobalAvgPool(nn.Module):
    def forward(self, inr):
        return inr.mean()
