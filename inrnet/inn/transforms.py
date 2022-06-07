"""Data augmentation"""
import torch
import numpy as np
nn=torch.nn
F=nn.functional

def intensity_noise(values:torch.Tensor, scale:float=.01) -> torch.Tensor:
    """Add Gaussian noise to values at each coordinate"""
    return values + torch.randn_like(values) * scale
    
def coord_noise(coords:torch.Tensor, scale:float=.01) -> torch.Tensor:
    """Add Gaussian noise to coordinate positions"""
    return coords + torch.randn_like(coords) * scale
    
def rand_flip(coords:torch.Tensor, axis:int=1, p:float=.5, domain:tuple=(-1,1)):
    """Randomly flip coordinates. Only works on symmetric domain."""
    assert domain==(-1,1)
    if np.random.rand() < p:
        coords[:, axis] = -coords[:, axis]
    # if np.random.rand() < p/10:
    #     coords[:, 0] = -coords[:, 0]
    return coords
    
# def rand_affine(coords, matrix):
#     return coords + torch.randn_like(coords) * scale
