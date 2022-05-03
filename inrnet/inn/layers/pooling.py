from functools import partial
import torch, pdb
nn = torch.nn
F = nn.functional

from inrnet.inn import functional as inrF

def translate_pool(layer, input_shape, extrema):
    if isinstance(layer, nn.MaxPool2d):
        h,w = input_shape
        extrema_dists = extrema[0][1] - extrema[0][0], extrema[1][1] - extrema[1][0]
        spacing = extrema_dists[0] / (h-1), extrema_dists[1] / (w-1)
        # k = layer.kernel_size * spacing[0], layer.kernel_size * spacing[1]
        s = layer.stride * spacing[0], layer.stride * spacing[1]
        if layer.kernel_size % 2 == 0:
            shift = spacing[0]/2, spacing[1]/2
        else:
            shift = 0,0
            
        out_shape = h//2, w//2
        new_extrema = ((extrema[0][0], extrema[0][1] - spacing[0]),
                    (extrema[1][0], extrema[1][1] - spacing[1]))
        if isinstance(layer.kernel_size, int):
            k = layer.kernel_size**2
        else:
            k = layer.kernel_size[0] * layer.kernel_size[1]
        return MaxPool(k=k, stride=s, shift=shift), out_shape, new_extrema
        # return MaxPoolKernel(kernel_size=k, stride=s, shift=shift), out_shape, new_extrema
    else:
        raise NotImplementedError


def max_pool_kernel(values, inr, layer, query_coords=None):
    coords = inr.sampled_coords
    if query_coords is None:
        if layer.stride != 0:
            inr.sampled_coords = query_coords = subsample_points_by_grid(coords, spacing=layer.stride)
        else:
            query_coords = coords

    if hasattr(layer, "norm"):
        Diffs = query_coords.unsqueeze(1) - coords.unsqueeze(0)
        mask = layer.norm(Diffs) < layer.radius
    else:
        if torch.amax(layer.shift) > 0:
            query_coords = query_coords + layer.shift
        Diffs = query_coords.unsqueeze(1) - coords.unsqueeze(0)
        mask = (Diffs[...,0].abs() < layer.kernel_size[0]/2) * (Diffs[...,1].abs() < layer.kernel_size[1]/2)
    lens = tuple(mask.sum(1))
    Y = values[torch.where(mask)[1]]
    Ysplit = Y.split(lens)
    return torch.stack([y.max(0).values for y in Ysplit], dim=0)

#((Diffs[:,-1,0].abs() <= layer.kernel_size[0]) * (Diffs[:,-1,1].abs() <= layer.kernel_size[1])).sum()




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
        new_inr.set_integrator(inrF.avg_pool, 'AvgPool', layer=self)
        return new_inr

class MaxPool(nn.Module):
    def __init__(self, k, stride=0., shift=(0,0)):
        super().__init__()
        self.k = k
        self.stride = stride
        self.register_buffer('shift', torch.tensor(shift))
    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.set_integrator(inrF.max_pool, 'MaxPool', layer=self)
        return new_inr

class MaxPoolKernel(nn.Module):
    def __init__(self, kernel_size, stride=0., shift=(0,0)):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.register_buffer('shift', torch.tensor(shift))
    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.set_integrator(max_pool_kernel, 'MaxPoolKernel', layer=self)
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
        new_inr.set_integrator(max_pool_kernel, 'MaxPoolBall', layer=self)
        return new_inr

class GlobalAvgPoolSequence(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    def forward(self, inr):
        vvf = inr.create_derived_inr()
        vvf.integrator = inrF.Integrator(GAPseq, 'GlobalPoolSeq', layer=self, inr=inr)
        return vvf

def GAPseq(values, layer, inr):
    if inr.training:
        layer.train()
    else:
        layer.eval()
    return layer.layers(values.mean(0, keepdim=True).float())
# class GlobalAvgPool(nn.Module):
#     def __init__(self, sampling="input"):
#         super().__init__()
#         self.sampling = sampling
#     def forward(self, inr, coords=None):
#         if coords is None:
#             if self.sampling == "qmc":
#                 coords = inr.generate_sample_points()
#                 return inrF.global_avg_pool(inr(coords))
#             elif self.sampling == "input":
#                 return inrF.global_avg_pool(inr(coords))

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
