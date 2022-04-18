from functools import partial
import torch
nn = torch.nn
F = nn.functional

from inrnet.inn import functional as inrF

class AvgPool(nn.Module):
    def __init__(self, radius, p_norm=2, stride=0.):
        super().__init__()
        self.radius = radius
        self.norm = partial(torch.linalg.norm, ord=p_norm, dim=-1)
    def forward(self, inr):
        inr.integrator = partial(inrF.avg_pool, layer=self)
        return inr.create_derived_inr()

class MaxPool(nn.Module):
    def __init__(self, radius, p_norm=2, stride=0.):
        super().__init__()
        self.radius = radius
        self.norm = partial(torch.linalg.norm, ord=p_norm, dim=-1)
    def forward(self, inr):
        inr.integrator = partial(inrF.max_pool, layer=self)
        return inr.create_derived_inr()

class GlobalAvgPool(nn.Module):
    def forward(self, inr):
        return inr.create_VVF(inrF.global_avg_pool)
