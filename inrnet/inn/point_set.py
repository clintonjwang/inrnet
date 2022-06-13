"""Point set"""
import itertools

import numpy as np
from inrnet.inn.support import BoundingBox, Support
from scipy.stats import qmc
import math
import torch
nn=torch.nn
from inrnet import util

PointValues = torch.Tensor
# class PointValues(torch.Tensor):
#     def batch_size(self):
#         return self.size(0)
#     def N(self):
#         return self.size(-2)
#     def channels(self):
#         return self.size(-1)
    
class PointSet(torch.Tensor):
    def N(self):
        return self.size(-2)
    def dims(self):
        return self.size(-1)
    # def estimate_discrepancy():
    #     return NotImplemented

def get_sampler_from_args(dl_args, c2f:bool=True):
    if dl_args['sample type'] == 'grid':
        sampler = {'sample type': 'grid', 'dims': dl_args['image shape'], 'c2f': c2f}
    else:
        sampler = {'sample type': dl_args['sample type'],
            'sample points': dl_args['sample points']}
    return sampler

def generate_sample_points(domain: Support, sampler: dict, **kwargs) -> PointSet:
    """Generates sample points for integrating along the INR

    Args:
        domain (INRBatch): domain of INR to sample
        sampler (Optional[str], optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: coordinates to sample
    """
    method = sampler['sample type']

    if method == "grid":
        assert 'dims' in sampler and 'c2f' in sampler
        coords = meshgrid_coords(*sampler['dims'], c2f=sampler['c2f'])

    elif method == "shrunk":
        assert 'dims' in sampler
        coords = gen_LD_seq_bbox(
            n=sampler['sample points'],
            bbox=domain.bounds, scramble=True)
        coords = coords * coords.abs()

    elif method in ("qmc", 'rqmc'):
        coords = gen_LD_seq_bbox(
            n=sampler['sample points'],
            bbox=domain.bounds, scramble=(method=='rqmc'))

    elif method == 'masked':
        assert 'mask' in kwargs
        coords = gen_LD_seq_bbox(
            n=sampler['sample points'],
            bbox=domain.bounds, scramble=(method=='rqmc'))

    else:
        raise NotImplementedError("invalid method: "+method)
    coords.sample_mode = method

    return coords


def gen_LD_seq_bbox(n:int, bbox:tuple[tuple[float]],
        scramble:bool=False, like=None, dtype=torch.float,
        device="cuda") -> PointSet:
    """Generates a low discrepancy point set on an orthotope.

    Args:
        n (int): number of points to generate.
        bbox (tuple, optional): bounds of domain
        scramble (bool, optional): randomized QMC. Defaults to False.
        like (_type_, optional): _description_. Defaults to None.
        dtype (_type_, optional): _description_. Defaults to torch.float.
        device (str, optional): _description_. Defaults to "cuda".

    Returns:
        PointSet (n,d): coordinates
    """
    d = len(bbox)
    assert d == 2, '2D only'
    if math.log2(n) % 1 == 0:
        sampler = qmc.Sobol(d=d, scramble=scramble)
        sample = sampler.random_base2(m=int(math.log2(n)))
    else:
        sampler = qmc.Halton(d=d, scramble=scramble)
        sample = sampler.random(n=n)
    if like is None:
        out = torch.as_tensor(sample, dtype=dtype, device=device)
    else:
        out = torch.as_tensor(sample, dtype=like.dtype, device=like.device)
    out[:,0] = out[:,0] * (bbox[0][1]-bbox[0][0]) + bbox[0][0]
    out[:,1] = out[:,1] * (bbox[1][1]-bbox[1][0]) + bbox[1][0]
    return out.as_subclass(PointSet)


def meshgrid_coords(*dims, domain=(-1,1), c2f=True,
        dtype=torch.float, device="cuda") -> PointSet:
    # c2f: coarse-to-fine ordering, puts points along coarser grid-points first
    tensors = [torch.linspace(*domain, steps=d, dtype=dtype, device=device) for d in dims]
    mgrid = torch.stack(util.meshgrid(*tensors, indexing='ij'), dim=-1)
        
    if c2f:
        x_indices = [0]
        y_indices = [0]
        factor = 2
        x_step = dims[0]//2
        y_step = dims[1]//2
        ind_iters = []
        while x_step > 0 or y_step > 0:
            if y_step > 0:
                if y_step > 1 and y_step % 2 == 1:
                    raise NotImplementedError('meshgrid is only working for powers of 2')
                    new_y_indices = [y for y in range(1,dims[1]) if y not in y_indices]
                    ind_iters += list(itertools.product(x_indices, new_y_indices))
                else:
                    new_y_indices = list(y_step * np.arange(1,factor,2))
                    ind_iters += list(itertools.product(x_indices, new_y_indices))

            if x_step > 0:
                if x_step > 1 and x_step % 2 == 1:
                    new_x_indices = [x for x in range(1,dims[0]) if x not in x_indices]
                    ind_iters += list(itertools.product(new_x_indices, y_indices))
                    x_step = 0
                else:
                    new_x_indices = list(x_step * np.arange(1,factor,2))
                    ind_iters += list(itertools.product(new_x_indices, y_indices))

                if y_step > 0:
                    ind_iters += list(itertools.product(new_x_indices, new_y_indices))
                x_indices += new_x_indices
                x_step = x_step//2
                
            if y_step > 0:
                if y_step > 1 and y_step % 2 == 1:
                    y_step = 0
                y_indices += new_y_indices
                y_step = y_step//2

            factor *= 2

        flat_grid = mgrid.reshape(-1, len(dims))
        indices = torch.tensor([(0,0),*ind_iters], device=device)
        indices = indices[:,0]*dims[1] + indices[:,1]
        coords = flat_grid[indices]

    else:
        coords = mgrid.reshape(-1, len(dims))
    return coords.as_subclass(PointSet)


def generate_masked_sample_points(mask: torch.Tensor, sample_size: int,
    eps:float=1/32) -> PointSet:
    """Generates a low discrepancy point set within a masked region.

    Args:
        mask (1,H,W): generated points must fall in this mask
        sample_size (int): number of points to generate
        eps (float, optional): _description_. Defaults to 1/32.

    Returns:
        tensor (N,d): sampled coordinates
    """
    mask = mask.squeeze()
    if len(mask.shape) != 2:
        raise NotImplementedError('2D only')

    H,W = mask.shape
    fraction = (mask.sum()/torch.numel(mask)).item()
    coords = gen_LD_seq_bbox(n=int(sample_size/fraction * 1.2),
                bbox=BoundingBox((eps,H-eps),(eps,W-eps)), scramble=True)
    coo = torch.floor(coords).long()
    bools = mask[coo[:,0], coo[:,1]]
    coord_subset = coords[bools]
    if coord_subset.size(0) < sample_size:
        return generate_masked_sample_points(mask, int(sample_size*1.5))
    return coord_subset[:sample_size]


def get_low_discrepancy_sequence_ball(N, radius=1., eps=0., dtype=torch.float, device="cuda"):
    # what we really want is a Voronoi partition that minimizes the
    # difference between the smallest and largest cell volumes, and includes (0,0)
    #
    # Fibonacci lattice
    # http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/#more-3069
    indices = torch.arange(0, N).to(device=device, dtype=dtype) + eps
    R = radius*(indices/(N-1+2*eps)).sqrt() * torch.sigmoid(torch.tensor(N).pow(.4))
    # shrink radius by some amount to increase Voronoi cells of outer points
    theta = torch.pi * (1 + 5**0.5) * indices
    return torch.stack((R*torch.cos(theta), R*torch.sin(theta)), dim=1)
