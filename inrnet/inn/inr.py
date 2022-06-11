"""Class for minibatches of INRs"""
from __future__ import annotations
from copy import copy
import operator
import pdb
from typing import Callable
import torch
from inrnet.inn.layers.merge import MergeLayer
from inrnet.inn.layers.other import PointWiseFunction
from inrnet.inn.point_set import PointSet, PointValues
from inrnet.inn.support import Support
nn=torch.nn
F=nn.functional

from inrnet import util

class INRBatch(nn.Module):
    """Standard INR minibatch"""
    def __init__(self, channels: int|None,
        domain: Support|None=None,
        device='cuda'):
        """
        Args:
            channels (int): output size
            domain (Support, optional): INR domain.
            device (str, optional): Defaults to 'cuda'.
        """
        super().__init__()
        self.channels = channels
        self.domain = domain
        self.detached = False
        self.device = device
        self.layer = None

    # @classmethod
    # def from_layer(cls, inr, layer):
    #     inr.inet()
    #     return cls(channels=layer.out_channels,
    #         input_dims=inr.input_dims,
    #         domain=inr.domain,
    #         device=inr.device)
        
    def forward(self, coords: PointSet) -> PointValues:
        return self.inet.evaluate(self.input_inr, coords)

    def __neg__(self):
        self.pointwise_transform(lambda x: -x, 'negation')
        return self

    def __add__(self, other):
        if isinstance(other, INRBatch):
            self.inet.append_layer(
                MergeLayer(self, other, merge_function=operator.__add__),
                parent_layers=(self.layer, other.layer),
            )
            return self.inet.get_layer_output()
        return self.copy_with_transform(lambda x: x+other, 'add')
    def __iadd__(self, other):
        self.pointwise_transform(lambda x: x+other, 'add')
        return self

    def __sub__(self, other):
        if isinstance(other, INRBatch):
            return MergeLayer(self, -other)
        return self.copy_with_transform(lambda x: x-other, 'subtract')
    def __isub__(self, other):
        self.pointwise_transform(lambda x: x-other, 'subtract')
        return self

    def __mul__(self, other):
        if isinstance(other, INRBatch):
            self.inet.append_layer(
                MergeLayer(self, other, merge_function=operator.__mul__),
                parent_layers=(self.layer, other.layer),
            )
            return self.inet.get_layer_output()
        return self.copy_with_transform(lambda x: x*other, 'multiply')
    def __imul__(self, other):
        self.pointwise_transform(lambda x: x*other, 'multiply')
        return self

    def __truediv__(self, other):
        if isinstance(other, INRBatch):
            self.inet.append_layer(
                MergeLayer(self, other, merge_function=operator.__truediv__),
                parent_layers=(self.layer, other.layer),
            )
            return self.inet.get_layer_output()
        return self.copy_with_transform(lambda x: x/other, 'divide')
    def __itruediv__(self, other):
        self.pointwise_transform(lambda x: x/other, 'divide')
        return self
    def __rtruediv__(self, other):
        return self.copy_with_transform(lambda x: other/x, 'reciprocal')
        # self.pointwise_transform(lambda x: other/x)
        # return self
    
    def cat(self, other):
        if isinstance(other, INRBatch):
            self.inet.append_layer(
                MergeLayer(self, other, merge_function=lambda x: torch.cat(x,other)),
                parent_layers=(self.layer, other.layer),
            )
            return self.inet.get_layer_output()
        else:
            return self.copy_with_transform(lambda x: torch.cat(x,other), 'concat')

    def copy_with_transform(self, modification: Callable, name: str) -> INRBatch:
        new_inr = self.create_derived_inr()
        new_inr.pointwise_transform(modification, name)
        return new_inr
    def create_derived_inr(self):
        return copy.copy(self)

    def pointwise_transform(self, modification: Callable, name: str) -> None:
        # if self.layer:
        new_layer = self.inet.append_layer(PointWiseFunction(modification), name)
        self.layer = new_layer
    def add_layer(self, layer: nn.Module) -> None:
        self.inet.add_layer(layer)
        # test_output = modification(torch.randn(1,self.channels).cuda()) #.double()
        # self.channels = test_output.size(1)

    def detach(self):
        self.detached = True
        return self

    def produce_images(self, H,W, dtype=torch.float):
        with torch.no_grad():
            xy_grid = util.meshgrid_coords(H,W, device=self.device)
            output = self.forward(xy_grid)
            output = util.realign_values(output, inr=self)
            output = output.reshape(output.size(0),H,W,-1)
        if dtype == 'numpy':
            return output.squeeze(-1).cpu().float().numpy()
        else:
            return output.permute(0,3,1,2).to(dtype=dtype).as_subclass(PointValues)



class BlackBoxINR(INRBatch):
    """
    Wrapper for arbitrary INR architectures (SIREN, NeRF, etc.).
    Not batched - this generates each INR one at a time, then concats them.
    """
    def __init__(self, evaluator, channels, **kwargs):
        super().__init__(channels=channels, **kwargs)
        self.evaluator = nn.ModuleList(evaluator).eval()
        self.spatial_transforms = []
        self.intensity_transforms = []

    def __repr__(self):
        return f"""BlackBoxINR(batch_size={len(self.evaluator)}, channels={self.channels}, modifiers={self.modifiers})"""

    def produce_images(self, H:int,W:int, dtype=torch.float):
        with torch.no_grad():
            xy_grid = util.meshgrid_coords(H,W, c2f=False, device=self.device)
            output = self.forward(xy_grid)
            output = output.reshape(output.size(0),H,W,-1)
        if dtype == 'numpy':
            return output.squeeze(-1).cpu().float().numpy()
        else:
            return output.permute(0,3,1,2).to(dtype=dtype).as_subclass(PointValues)

    def add_transforms(self, spatial=None, intensity=None) -> None:
        if spatial is not None:
            if not hasattr(spatial, '__iter__'):
                spatial = [spatial]
            self.spatial_transforms += spatial
        if intensity is not None:
            if not hasattr(intensity, '__iter__'):
                intensity = [intensity]
            self.intensity_transforms += intensity

    def forward(self, coords: PointSet) -> PointValues:
        if hasattr(self, "cached_outputs") and self.sampled_coords.shape == coords.shape and torch.allclose(self.sampled_coords, coords):
            return self.cached_outputs
        with torch.no_grad():
            for tx in self.spatial_transforms:
                coords = tx(coords)
            self.sampled_coords = coords
            out = []
            for inr in self.evaluator:
                out.append(inr(coords))
            out = torch.stack(out, dim=0).as_subclass(PointValues)
            if len(out.shape) == 4:
                out.squeeze_(0)
                if len(out.shape) == 4:
                    raise ValueError('bad BBINR evaluator')
            for tx in self.intensity_transforms:
                out = tx(out)
        for m in self.modifiers:
            out = m(out)
        self.cached_outputs = out
        return out

    def forward_with_grad(self, coords: PointSet) -> PointValues:
        self.sampled_coords = coords
        out = []
        for inr in self.evaluator:
            out.append(inr(coords))
        out = torch.stack(out, dim=0).as_subclass(PointValues)
        for m in self.modifiers:
            out = m(out)
        return out
