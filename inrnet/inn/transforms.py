"""Data augmentation"""
import torch
import numpy as np

from inrnet.inn.point_set import PointSet, PointValues
nn=torch.nn
F=nn.functional

def intensity_noise(values: PointValues, scale:float=.01) -> PointValues:
    """Add Gaussian noise to values at each coordinate"""
    return values + torch.randn_like(values) * scale
    
def coord_noise(coords: PointSet, scale:float=.01) -> PointSet:
    """Add Gaussian noise to coordinate positions"""
    return coords + torch.randn_like(coords) * scale
    
def rand_flip(coords: PointSet, axis: int=1, p: float=.5, domain: tuple=(-1,1)) -> PointSet:
    """Randomly flip coordinates. Only works on symmetric domain."""
    assert domain==(-1,1)
    if np.random.rand() < p:
        coords[:, axis] = -coords[:, axis]
    return coords
    
def vertical_flip(coords: PointSet, p: float=.5, domain: tuple=(-1,1)) -> PointSet:
    return rand_flip(coords, axis=0, p=p, domain=domain)
def horizontal_flip(coords: PointSet, p: float=.5, domain: tuple=(-1,1)) -> PointSet:
    return rand_flip(coords, axis=1, p=p, domain=domain)
    
# def rand_affine(coords, matrix):
#     return coords + torch.randn_like(coords) * scale
