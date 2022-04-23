from functools import partial
import torch
nn = torch.nn
F = nn.functional

from inrnet.inn import functional as inrF

def translate_pool(layer, input_shape, extrema):
    if isinstance(layer, nn.MaxPool2d):
        h,w = input_shape
        extrema_dists = extrema[0][1] - extrema[0][0], extrema[1][1] - extrema[1][0]
        spacing = extrema_dists[0] / (h-1), extrema_dists[1] / (w-1)
        k = layer.kernel_size * spacing[0], layer.kernel_size * spacing[1]
        s = layer.stride * spacing[0], layer.stride * spacing[1] # add a little extra to prevent boundary effects
        if layer.kernel_size % 2 == 0:
            shift = k[0]/4, k[1]/4
        else:
            shift = 0,0
        out_shape = h//2, w//2
        new_extrema = ((extrema[0][0], extrema[0][1] - spacing[0]),
                    (extrema[1][0], extrema[1][1] - spacing[1]))
        return MaxPool(kernel_size=k, stride=s, shift=shift), out_shape, new_extrema
    else:
        raise NotImplementedError

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
    def __init__(self, kernel_size, stride=0., shift=0.):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.shift = shift
    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.integrator = partial(inrF.max_pool, inr=new_inr, layer=self)
        return new_inr

class MaxPoolBall(nn.Module):
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

class AdaptiveAvgPool(nn.Module):
    def __init__(self, output_size, p_norm="inf"):
        super().__init__()
        self.output_size = output_size
        if p_norm == "inf":
            p_norm = torch.inf
        self.norm = partial(torch.linalg.norm, ord=p_norm, dim=-1)
    def forward(self, inr):
        coords = inr.generate_sample_points()
        return inrF.adaptive_avg_pool(inr(coords))
