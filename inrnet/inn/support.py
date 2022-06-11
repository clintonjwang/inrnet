import torch

class Support:
    def in_support(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented

class BoundingBox(Support):
    def __init__(self, ranges: tuple[tuple[float]]):
        """
        Args:
            ranges (tuple[tuple[float]]): must be in format ((0,1),(0,1))
        """        
        self.ranges = ranges
        self.dimensionality = len(ranges)
        assert len(ranges) == 2 and len(ranges[0]) == 2
        
    def in_support(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x[...,0] > self.ranges[0][0]) * (x[...,0].abs() < self.ranges[0][0]) * (
            x[...,1] > self.ranges[1][0]) * (x[...,1].abs() < self.ranges[1][0]
        )
    
class Orthotope(Support):
    def __init__(self, dims: tuple[float], grid_shift: tuple[float]=(0,0)):
        """
        Args:
            dims (tuple[float]): Dimensions.
            grid_shift (tuple[int], optional): grid_shift of the kernel. Defaults to (0,0).
        """        
        # if not hasattr(dims, '__iter__'):
        #     dims = (dims, dims)
        self.dims = dims
        self.dimensionality = len(dims)
        self.grid_shift = torch.tensor(grid_shift)

    def in_support(self, x: torch.Tensor) -> torch.Tensor:
        return (x[...,0].abs() < self.dims[0]/2) * (x[...,1].abs() < self.dims[1]/2)

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