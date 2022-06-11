from __future__ import annotations
import typing
import numpy as np
import torch
from graphlib import TopologicalSorter

from inrnet.inn.point_set import PointSet, PointValues
from inrnet.inn.support import Support

if typing.TYPE_CHECKING:
    from inrnet.inn.inr import INRBatch
nn = torch.nn

from inrnet import inn, util

class INRNet(nn.Module):
    def __init__(self, domain: Support|None=None,
        sample_mode='qmc', sample_size=256, layers=None):
        """
        Args:
            domain (Support, optional): domain of valid INRs.
            device (str, optional): Defaults to 'cuda'.
        """
        super().__init__()
        self.domain = domain
        self.sample_mode = sample_mode
        self.sample_size = sample_size
        self.layers = {}
        self.parents = {}
        self.children = {}
        if layers is None:
            self.output_layer = self.current_layer = None
        else:
            assert isinstance(layers, tuple) or isinstance(layers, list)
            prev_name = None
            for layer in layers:
                if name is None:
                    name = layer.name+".0"
                while name in self.layers:
                    name = util.increment_name(name)
                self.layers[name] = layer
                if prev_name is not None:
                    self.parents[name] = [prev_name]
                self.children = {}
            self.output_layer = self.current_layer = layer

    @property
    def volume(self):
        if isinstance(self.domain, tuple):
            return (self.domain[1] - self.domain[0])**self.input_dims
        else:
            return np.prod([d[1]-d[0] for d in self.domain])
        
    def insert_layer(self, layer: nn.Module, name: str=None,
        parent_layers=None, in_between=True):
        """Insert a single layer.
        Makes it a child of the current layer, unless parents are specified.
        Set the INR-Net's current layer to this new layer.

        Args:
            layer (Module): layer to add
            name (str, optional): name of new layer
            parent_layers (Modules, optional): name of new layer
            in_between (bool, optional): By default, changes existing children
                of the current layer to children of the new layer. If False,
                does not affect the existing children of the current layer.
        """
        if name is None:
            name = layer.name+".0"
        while name in self.layers:
            name = util.increment_name(name)
        self.layers[name] = layer
        self.children[name] = []

        if parent_layers is None:
            if self.current_layer is not None:
                parent_layers = [self.current_layer]
        elif not (isinstance(parent_layers, tuple) or isinstance(parent_layers, list)):
            parent_layers = [parent_layers]
        parent_layers = [p for p in parent_layers if p is not None]
        if parent_layers is None or len(parent_layers) == 0:
            self.current_layer = name
            self.parents[name] = []
            return

        if in_between is True:
            for p in parent_layers:
                for child in self.children[p]:
                    assert len(self.parents[child]) == 1
                    self.parents[child] = [name]
                self.children[p] = []

        for parent_layer in parent_layers:
            self.children[parent_layer].append(name)
        self.parents[name] = parent_layers

        self.current_layer = name
        
    def forward(self, inr: INRBatch) -> INRBatch:
        if self.output_layer is None:
            self.output_layer = self.current_layer
        return INRBatch.from_layer(self.output_layer)

    def inr_forward(self, coords: PointSet) -> PointValues:
        if self.detached:
            if hasattr(self, "cached_outputs"):
                return self.cached_outputs.detach()
            with torch.no_grad():
                return self._forward(coords)
        return self._forward(coords)

    def _forward(self, coords: PointSet) -> PointValues:
        if hasattr(self, "cached_outputs"):
            return self.cached_outputs

        out = self.inet(coords)
        self.sampled_coords = self.inet.sampled_coords
        if hasattr(self.inet, 'dropped_coords'):
            self.dropped_coords = self.inet.dropped_coords
        
        self.inet(out)
        for m in self.modifiers:
            out = m(out)
        if self.caching_enabled:
            self.cached_outputs = out
        return out

    # def forward(self, inr: INRBatch, coords: PointSet|None=None):
    #     return self.sequential(inr)
        # ts = TopologicalSorter(graph)
        # layer_order = tuple(ts.static_order())

class VectorValuedINRNet(INRNet):
    def forward(self, inr: INRBatch, coords: PointSet) -> PointValues:
        if self.detached:
            if hasattr(self, "cached_outputs"):
                return self.cached_outputs.detach()
            with torch.no_grad():
                return self._forward(coords)
        return self._forward(coords)

            
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


# class CondLayer(INRBatch):
#     def __init__(self, channels: int, cond_integrator=False,
#         input_dims: int=2, domain: Tuple[int]=(-1,1)):
#         super().__init__(channels, input_dims, domain)
#         self.cond_integrator = cond_integrator

#     def forward(self, coords: PointSet, condition: Callable) -> PointValues:
#         if self.detached:
#             if hasattr(self, "cached_outputs"):
#                 return self.cached_outputs.detach()
#             with torch.no_grad():
#                 return self._forward(coords, condition)
#         else:
#             return self._forward(coords, condition)

#     def _forward(self, coords: PointSet, condition: Callable) -> PointValues:
#         if hasattr(self, "cached_outputs"):
#             return self.cached_outputs
#         if isinstance(self.evaluator, CondINR):
#             out = self.evaluator(coords, condition)
#         else:
#             out = self.evaluator(coords)
#         # try:
#         self.sampled_coords = self.evaluator.sampled_coords
#         # except AttributeError:
#         #     self.sampled_coords = self.origin.sampled_coords
#         if self.integrator is not None:
#             if self.cond_integrator:
#                 out = self.integrator(out, condition)
#             else:
#                 out = self.integrator(out)
#         for m in self.modifiers:
#             out = m(out)
#         if self.caching_enabled:
#             self.cached_outputs = out
#         return out
