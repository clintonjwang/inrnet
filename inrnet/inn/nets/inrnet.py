from __future__ import annotations
import typing
import numpy as np
import torch
from graphlib import TopologicalSorter

from inrnet.inn.point_set import PointSet

if typing.TYPE_CHECKING:
    from inrnet.inn.inr import INRBatch
nn = torch.nn

from inrnet import inn, util

class INRNet(nn.Module):
    def __init__(self, sequential, domain=(-1,1), sample_mode='qmc', sample_size=256):
        super().__init__()
        self.domain = domain
        self.sample_mode = sample_mode
        self.sample_size = sample_size
        self.current_layer = None
        self.sequential = sequential
        # self.layers = {}

    @property
    def volume(self):
        if isinstance(self.domain, tuple):
            return (self.domain[1] - self.domain[0])**self.input_dims
        else:
            return np.prod([d[1]-d[0] for d in self.domain])

    def append_layer(self, layer):
        name = layer.name+".0"
        while name in self.layers:
            name = util.increment_name(name)
        if self.current_layer is not None:
            self.layers[self.current_layer]['successors'].append(name)
        self.layers[name] = {'layer': layer, 'successors': []}
        self.current_layer = name

    # def insert_layer(self, layer, predecessor):
    #     self.current_layer = predecessor
    #     self.layers['successors']
    #     self.append_layer(layer)

    def forward(self, inr: INRBatch, coords: PointSet|None=None):
        return self.sequential(inr)
        # ts = TopologicalSorter(graph)
        # layer_order = tuple(ts.static_order())

            
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

