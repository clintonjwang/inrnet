from math import gamma
import numpy as np
import torch

from inrnet.inn.point_set import PointSet

class Support:
    def in_support(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented
        
    @property
    def volume(self):
        return NotImplemented
        
    def sample_points(self, N: int, method: str) -> PointSet:
        return NotImplemented

class BoundingBox(Support):
    def __init__(self, bounds: tuple[tuple[float]]):
        """Constructor from bounds.

        Args:
            bounds (tuple[tuple[float]]): must be in format ((0,1),(0,1))
        """        
        self.bounds = bounds
        self.dimensionality = len(bounds)
        assert len(bounds) == 2 and len(bounds[0]) == 2
    
    @classmethod
    def from_orthotope(cls, dims: tuple[float], center: tuple[float]=(0,0)):
        """Alternative constructor by specifying shape and center.

        Args:
            dims (tuple[float]): Dimensions.
            center (tuple[int], optional): center of the kernel. Defaults to (0,0).
        """
        bounds = [(dims[ix][0] + center[0],
            dims[ix][0] + center[1]) for ix in range(len(dims))]
        return cls(bounds)

    def in_support(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x[...,0] > self.bounds[0][0]) * (x[...,0] < self.bounds[0][1]) * (
            x[...,1] > self.bounds[1][0]) * (x[...,1] < self.bounds[1][1]
        )

    @property
    def volume(self):
        return np.prod([r[1]-r[0] for r in self.ranges])

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