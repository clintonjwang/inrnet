from functools import partial
import torch
nn = torch.nn
F = nn.functional

from inrnet.inn import functional as inrF

def translate_pool(layer, img_shape):
    if isinstance(layer, nn.MaxPool2d):
        h,w = img_shape
        k = layer.kernel_size
        s = layer.stride
        K = k / h, k / w
        return MaxPool(kernel_size=K)
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

class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride=0.):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
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
