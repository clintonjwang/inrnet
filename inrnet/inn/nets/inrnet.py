from __future__ import annotations
import typing
import numpy as np
import torch
from inrnet.inn.inr import DiscretizedINR
from inrnet.inn.point_set import PointValues, generate_sample_points
from inrnet.inn.support import Support

if typing.TYPE_CHECKING:
    from inrnet.inn.inr import INRBatch
nn = torch.nn

from inrnet import inn, util

class INRNet(nn.Module):
    def __init__(self, sampler: dict, layers=None):
        """
        Args:
            domain (Support, optional): valid INR domain
        """
        super().__init__()
        self.sampler = sampler
        self.layers = layers

    def __len__(self):
        return len(self.layers)
    def __iter__(self):
        return iter(self.layers)
    def __getitem__(self, ix):
        return self.layers[ix]

    def sample_inr(self, inr: INRBatch) -> DiscretizedINR:
        coords = generate_sample_points(domain=inr.domain, sampler=self.sampler)
        return DiscretizedINR(coords, values=inr(coords), domain=inr.domain)
        
    def forward(self, inr: INRBatch) -> DiscretizedINR|torch.Tensor:
        d_inr = self.sample_inr(inr)
        return self.layers(d_inr)

    def produce_images(self, inr: INRBatch, H,W, dtype=torch.float):
        with torch.no_grad():
            xy_grid = util.meshgrid_coords(H,W, device=self.device)
            coords = generate_sample_points(domain=inr.domain, sampler={
                'sample type': 'grid', 'dims': (H,W),
            })
            d_inr = DiscretizedINR(coords, values=inr(coords), domain=inr.domain)
            out_inr = self.layers(d_inr)
            output = util.sort_inr(out_inr).values
            output = output.reshape(output.size(0),H,W,-1)
        if dtype == 'numpy':
            return output.squeeze(-1).cpu().float().numpy()
        else:
            return output.permute(0,3,1,2).to(dtype=dtype).as_subclass(PointValues)

def freeze_layer_types(inrnet, classes=(inn.ChannelMixer, inn.ChannelNorm)):
    for m in inrnet:
        if hasattr(m, '__iter__'):
            freeze_layer_types(m, classes)
        elif m.__class__ in classes:
            for param in m.parameters():
                param.requires_grad = False

def unfreeze_layer_types(inrnet, classes=(inn.ChannelMixer, inn.ChannelNorm)):
    for m in inrnet:
        if hasattr(m, '__iter__'):
            unfreeze_layer_types(m, classes)
        elif m.__class__ in classes:
            for param in m.parameters():
                param.requires_grad = True

def replace_conv_kernels(inrnet, k_type='mlp', k_ratio=1.5):
    length = len(inrnet)
    for i in range(length):
        m = inrnet[i]
        if hasattr(m, '__getitem__'):
            replace_conv_kernels(m, k_ratio=k_ratio)
        elif isinstance(m, inn.SplineConv):
            inrnet[i] = replace_conv_kernel(m, k_ratio=k_ratio)

def replace_conv_kernel(layer, k_type='mlp', k_ratio=1.5):
    #if k_type
    if isinstance(layer, inn.SplineConv):
        conv = inn.MLPConv(layer.in_channels, layer.out_channels, [k*k_ratio for k in layer.kernel_size],
            down_ratio=layer.down_ratio, groups=layer.groups)
        # conv.padded_extrema = layer.padded_extrema
        conv.bias = layer.bias
        return conv
    raise NotImplementedError
