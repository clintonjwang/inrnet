"""Pooling Layer"""
from functools import partial
import torch

from inrnet.inn.inr import INRBatch
nn = torch.nn
F = nn.functional

from inrnet.inn import functional as inrF

def translate_pool(layer, input_shape, extrema):
    h,w = input_shape
    extrema_dists = extrema[0][1] - extrema[0][0], extrema[1][1] - extrema[1][0]
    spacing = extrema_dists[0] / (h-1), extrema_dists[1] / (w-1)
    k = layer.kernel_size * spacing[0], layer.kernel_size * spacing[1]
    if layer.kernel_size % 2 == 0:
        shift = spacing[0]/2, spacing[1]/2
    else:
        shift = 0,0
        
    out_shape = h//2, w//2
    new_extrema = ((extrema[0][0], extrema[0][1] - spacing[0]),
                (extrema[1][0], extrema[1][1] - spacing[1]))
    if isinstance(layer, nn.MaxPool2d):
        return MaxPool(kernel_size=k, down_ratio=(1/layer.stride)**2, shift=shift), out_shape, new_extrema
    elif isinstance(layer, nn.AvgPool2d):
        return AvgPool(kernel_size=k, down_ratio=(1/layer.stride)**2, shift=shift), out_shape, new_extrema
    else:
        raise NotImplementedError
    # if isinstance(layer.kernel_size, int):
    #     k = layer.kernel_size**2
    # else:
    #     k = layer.kernel_size[0] * layer.kernel_size[1]
    # return MaxPoolKernel(kernel_size=k, down_ratio=s, shift=shift), out_shape, new_extrema


class AvgPool(nn.Module):
    def __init__(self, kernel_size, down_ratio=.25, shift=(0,0)):
        super().__init__()
        if not hasattr(kernel_size, '__iter__'):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.down_ratio = down_ratio
        self.register_buffer('shift', torch.tensor(shift))
        self.diffs_in_support = lambda diffs: (diffs[...,0].abs() < self.kernel_size[0]/2) * (
                        diffs[...,1].abs() < self.kernel_size[1]/2)

    def forward(self, inr: INRBatch) -> INRBatch:
        new_inr = inr.create_derived_inr()
        new_inr.set_integrator(inrF.avg_pool, 'AvgPool', layer=self)
        return new_inr

class AvgPoolNeighbor(nn.Module):
    def __init__(self, k, down_ratio=1., shift=(0,0)):
        super().__init__()
        self.num_neighbors = k
        self.down_ratio = down_ratio
        self.register_buffer('shift', torch.tensor(shift))
    def forward(self, inr: INRBatch) -> INRBatch:
        new_inr = inr.create_derived_inr()
        new_inr.set_integrator(inrF.avg_pool, 'AvgPool', layer=self)
        return new_inr

class AvgPoolBall(nn.Module):
    def __init__(self, radius, down_ratio=1., p_norm="inf"):
        super().__init__()
        self.radius = radius
        self.down_ratio = down_ratio
        if p_norm == "inf":
            p_norm = torch.inf
        self.norm = partial(torch.linalg.norm, ord=p_norm, dim=-1)
        self.diffs_in_support = lambda diffs: self.norm(diffs) < self.radius
        
    def forward(self, inr: INRBatch) -> INRBatch:
        new_inr = inr.create_derived_inr()
        new_inr.set_integrator(inrF.avg_pool, 'AvgPool', layer=self)
        return new_inr

class MaxPool(nn.Module):
    def __init__(self, kernel_size, down_ratio=.25, shift=(0,0)):
        super().__init__()
        if not hasattr(kernel_size, '__iter__'):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.down_ratio = down_ratio
        self.register_buffer('shift', torch.tensor(shift))
        self.diffs_in_support = lambda diffs: (diffs[...,0].abs() < self.kernel_size[0]/2) * (
                        diffs[...,1].abs() < self.kernel_size[1]/2)

    def forward(self, inr: INRBatch) -> INRBatch:
        new_inr = inr.create_derived_inr()
        new_inr.set_integrator(inrF.max_pool, 'MaxPool', layer=self)
        return new_inr

class MaxPoolNeighbor(nn.Module):
    def __init__(self, k, down_ratio=1., shift=(0,0)):
        super().__init__()
        self.num_neighbors = k
        self.down_ratio = down_ratio
        self.register_buffer('shift', torch.tensor(shift))
    def forward(self, inr: INRBatch) -> INRBatch:
        new_inr = inr.create_derived_inr()
        new_inr.set_integrator(inrF.max_pool, 'MaxPoolkNN', layer=self)
        return new_inr

class MaxPoolBall(nn.Module):
    def __init__(self, radius, down_ratio=1., p_norm="inf"):
        super().__init__()
        self.radius = radius
        self.down_ratio = down_ratio
        if p_norm == "inf":
            p_norm = torch.inf
        self.norm = partial(torch.linalg.norm, ord=p_norm, dim=-1)
        self.diffs_in_support = lambda diffs: self.norm(diffs) < self.radius
        
    def forward(self, inr: INRBatch) -> INRBatch:
        new_inr = inr.create_derived_inr()
        new_inr.set_integrator(inrF.max_pool, 'MaxPoolBall', layer=self)
        return new_inr
