from functools import partial
import torch
nn = torch.nn
F = nn.functional

from inrnet.inn import functional as inrF

class AvgPool(nn.Module):
    def __init__(self, radius, stride=0., p_norm="inf"):
        super().__init__()
        self.radius = radius
        self.stride = stride
        if p_norm == "inf":
            p_norm = torch.inf
        self.norm = partial(torch.linalg.norm, ord=p_norm, dim=-1)
    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.integrator = partial(inrF.avg_pool, inr=new_inr, layer=self)
        return new_inr

class MaxPool(nn.Module):
    def __init__(self, radius, stride=0., p_norm="inf"):
        super().__init__()
        self.radius = radius
        self.stride = stride
        if p_norm == "inf":
            p_norm = torch.inf
        self.norm = partial(torch.linalg.norm, ord=p_norm, dim=-1)
    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.integrator = partial(inrF.max_pool, inr=new_inr, layer=self)
        return new_inr

class GlobalAvgPool(nn.Module):
    def forward(self, inr):
        coords = inr.generate_sample_points()
        return inrF.global_avg_pool(inr(coords))
