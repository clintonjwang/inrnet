from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from inrnet.inn.point_set import PointSet

from math import gamma
import numpy as np
import torch

class Support:
    def in_support(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented
        
    @property
    def volume(self):
        return NotImplemented
        
    def sample_points(self, N: int, method: str) -> PointSet:
        return NotImplemented

class BoundingBox(Support):
    def __init__(self, *bounds):
        """Constructor from bounds.

        Args:
            bounds (tuple[tuple[float]]): must be in format (0,1),(0,1)
        """
        if len(bounds) == 1 and hasattr(bounds[0][0], '__iter__'):
            bounds = bounds[0]
        self.bounds = bounds
        self.dimensionality = len(bounds)
        assert len(bounds) == 2 and len(bounds[0]) == 2
        
    def __str__(self):
        return 'BoundingBox'
    def __repr__(self):
        return f'BoundingBox({self.bounds})'

    @classmethod
    def from_orthotope(cls, dims: tuple[float], center: tuple[float]|None=None):
        """Alternative constructor by specifying shape and center.

        Args:
            dims (tuple[float]): Dimensions.
            center (tuple[int], optional): center of the kernel. Defaults to (0,0).
        """
        if center is None:
            center = [0] * len(dims)
        bounds = [(-dims[ix]/2 + center[ix], dims[ix]/2 + center[ix]) for \
                                            ix in range(len(dims))]
        return cls(bounds)

    @property
    def shape(self):
        return [b[1]-b[0] for b in self.bounds]
    @property
    def volume(self):
        return np.prod(self.shape)

    def in_support(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x[...,0] > self.bounds[0][0]) * (x[...,0] < self.bounds[0][1]) * (
            x[...,1] > self.bounds[1][0]) * (x[...,1] < self.bounds[1][1]
        )

class Ball(Support):
    def __init__(self, radius: float, p_norm: str="inf",
        dimensionality: int=2):#, center: tuple[float]=(0,0)):
        """
        Args:
            radius (float)
            p_norm (str, optional): Defaults to "inf".
            dimensionality (int): Defaults to 2.
        """
        #center (tuple[int], optional): Center of the kernel. Defaults to (0,0).
        self.radius = radius
        self.p_norm = p_norm
        self.dimensionality = dimensionality

    def __str__(self):
        return 'Ball'
    def __repr__(self):
        return f'Ball({self.radius})'

    def in_support(self, x: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(x, ord=self.p_norm, dim=-1) < self.radius
    
    @property
    def volume(self):
        if self.p_norm == 'inf':
            return self.radius**self.dimensionality
        elif self.p_norm == 2:
            if self.dimensionality == 2:
                return np.pi * self.radius**2
            elif self.dimensionality == 3:
                return 4/3*np.pi * self.radius**3
            else:
                n = self.dimensionality
                return np.pi**(n/2) / gamma(n/2 + 1) * self.radius**n