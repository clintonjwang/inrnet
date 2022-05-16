import math, torch, pdb
import numpy as np
nn=torch.nn
F=nn.functional



def intensity_noise(values, scale=.005):
    return values + torch.randn_like(values) * scale
    
def coord_noise(coords, scale=.005):
    return coords + torch.randn_like(coords) * scale
    
def rand_flip(coords, axis=1, p=.5):
    if np.random.rand() < p:
        coords[:, axis] = -coords[:, axis]
    return coords
    
# def rand_affine(coords, matrix):
#     return coords + torch.randn_like(coords) * scale
