"""Class for minibatches of INRs"""
from __future__ import annotations
from copy import copy
import operator
import pdb
from typing import Callable
import torch
from inrnet.inn import point_set
from inrnet.inn.layers.merge import MergeLayer
from inrnet.inn.point_set import PointSet, PointValues
from inrnet.inn.support import Support
nn=torch.nn
F=nn.functional

class INRBatch(nn.Module):
    """Standard INR minibatch"""
    def __init__(self, channels: int|None=None,
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

    def forward(self, coords: PointSet) -> PointValues:
        return NotImplemented

    def create_derived_inr(self):
        return copy(self)


class DiscretizedINR(INRBatch):
    def __init__(self, coords: PointSet, values: PointValues, domain: Support|None=None):
        """INR represented as its points and values at those points

        Args:
            coords (PointSet): coordinates of sample points
            values (PointValues): values of sample points
            domain (Support, optional): INR domain.
        """
        super().__init__(channels=values.size(-1), domain=domain)
        self.register_buffer('coords', coords)
        self.register_buffer('values', values)
    
    def copy_with_transform(self, modification: Callable) -> DiscretizedINR:
        inr = self.create_derived_inr()
        inr.values = modification(self.values)
        return inr

    def sort(self):
        indices = torch.sort((self.coords[:,0]+2)*self.coords.size(0)/2 + self.coords[:,1]).indices
        self.values = self.values[:,indices]
        self.coords = self.coords[:,indices]

    def __neg__(self):
        self.values = -self.values
        return self

    def __add__(self, other):
        if isinstance(other, DiscretizedINR):
            return self.copy_with_transform(lambda x: x + other.values)
        return self.copy_with_transform(lambda x: x + other)
    def __iadd__(self, other):
        self.values += other
        return self

    def __sub__(self, other):
        if isinstance(other, DiscretizedINR):
            return self.copy_with_transform(lambda x: x - other.values)
        return self.copy_with_transform(lambda x: x - other)
    def __isub__(self, other):
        self.values -= other
        return self

    def __mul__(self, other):
        if isinstance(other, DiscretizedINR):
            return self.copy_with_transform(lambda x: x * other.values)
        return self.copy_with_transform(lambda x: x * other)
    def __imul__(self, other):
        self.values *= other
        return self

    def __truediv__(self, other):
        if isinstance(other, DiscretizedINR):
            return self.copy_with_transform(lambda x: x / other.values)
        return self.copy_with_transform(lambda x: x / other)
    def __itruediv__(self, other):
        self.values /= other
        return self
    def __rtruediv__(self, other):
        return self.copy_with_transform(lambda x: other/x)
    
    def matmul(self, other):
        if isinstance(other, DiscretizedINR):
            return self.copy_with_transform(lambda x: x.matmul(other.values))
        return self.copy_with_transform(lambda x: x.matmul(other))



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
        return f"""BlackBoxINR(batch_size={len(self.evaluator)}, channels={self.channels})"""

    def produce_images(self, H:int,W:int, dtype=torch.float):
        with torch.no_grad():
            xy_grid = point_set.meshgrid_coords(H,W, c2f=False, device=self.device)
            output = self.forward(xy_grid)
            output = output.reshape(output.size(0),H,W,-1)
        if dtype == 'numpy':
            return output.squeeze(-1).cpu().float().numpy()
        else:
            return output.permute(0,3,1,2).to(dtype=dtype)#.as_subclass(PointValues)

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
        if hasattr(self, "cached_outputs") and self.coords.shape == coords.shape and torch.allclose(self.coords, coords):
            return self.cached_outputs
        with torch.no_grad():
            for tx in self.spatial_transforms:
                coords = tx(coords)
            self.coords = coords
            out = []
            for inr in self.evaluator:
                out.append(inr(coords))
            out = torch.stack(out, dim=0)#.as_subclass(PointValues)
            if len(out.shape) == 4:
                out.squeeze_(0)
                if len(out.shape) == 4:
                    raise ValueError('bad BBINR evaluator')
            for tx in self.intensity_transforms:
                out = tx(out)
        self.cached_outputs = out
        return out

    def forward_with_grad(self, coords: PointSet) -> PointValues:
        self.coords = coords
        out = []
        for inr in self.evaluator:
            out.append(inr(coords))
        out = torch.stack(out, dim=0)#.as_subclass(PointValues)
        return out
