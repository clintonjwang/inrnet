"""Pooling Layer"""
import torch
from inrnet.inn.inr import DiscretizedINR
from inrnet.inn.support import BoundingBox, Support
nn = torch.nn
F = nn.functional

from inrnet.inn import functional as inrF

def translate_pool(layer, input_shape, extrema):
    h,w = input_shape
    extrema_dists = extrema[0][1] - extrema[0][0], extrema[1][1] - extrema[1][0]
    spacing = extrema_dists[0] / (h-1), extrema_dists[1] / (w-1)
    k = layer.kernel_size * spacing[0], layer.kernel_size * spacing[1]
    if layer.kernel_size % 2 == 0:
        grid_shift = spacing[0]/2, spacing[1]/2
    else:
        grid_shift = 0,0
        
    out_shape = h//2, w//2
    new_extrema = ((extrema[0][0], extrema[0][1] - spacing[0]),
                (extrema[1][0], extrema[1][1] - spacing[1]))
    if isinstance(layer, nn.MaxPool2d):
        layer_type = MaxPool
    elif isinstance(layer, nn.AvgPool2d):
        layer_type = AvgPool
    else:
        raise NotImplementedError
    return layer_type(support=BoundingBox.from_orthotope(dims=(k,k), grid_shift=grid_shift),
                down_ratio=(1/layer.stride)**2), out_shape, new_extrema
    # if isinstance(layer.kernel_size, int):
    #     k = layer.kernel_size**2
    # else:
    #     k = layer.kernel_size[0] * layer.kernel_size[1]
    # return MaxPoolKernel(kernel_size=k, down_ratio=s, shift=shift), out_shape, new_extrema

class KernelPool(nn.Module):
    def __init__(self, support: Support, down_ratio: float=.25):
        """Pooling layer with fixed-size kernel

        Args:
            support (Support): kernel support
            down_ratio (float, optional): Ratio of output points to input points. Defaults to .25.
        """
        super().__init__()
        self.support = support
        self.down_ratio = down_ratio

class AvgPool(KernelPool):
    def forward(self, inr: DiscretizedINR) -> DiscretizedINR:
        new_inr = inr.create_derived_inr()
        inrF.avg_pool('AvgPool', support=self.support, down_ratio=self.down_ratio)
        return new_inr

class MaxPool(KernelPool):
    def forward(self, inr: DiscretizedINR) -> DiscretizedINR:
        new_inr = inr.create_derived_inr()
        inrF.max_pool('MaxPool', support=self.support, down_ratio=self.down_ratio)
        return new_inr
