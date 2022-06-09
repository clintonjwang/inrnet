"""Point set"""
from __future__ import annotations
import typing
from scipy.stats import qmc
import math
import torch
nn=torch.nn

if typing.TYPE_CHECKING:
    from inrnet.inn.inr import INRBatch
from inrnet import util

class PointValues(torch.Tensor):
    def batch_size(self):
        return self.size(0)
    def N(self):
        return self.size(-2)
    def channels(self):
        return self.size(-1)
    
class PointSet(torch.Tensor):
    def N(self):
        return self.size(-2)
    def dims(self):
        return self.size(-1)
    # def estimate_discrepancy():
    #     return NotImplemented

def generate_sample_points(inr: INRBatch, dl_args: dict) -> PointSet:
    """Generates sample points for integrating along the INR

    Args:
        inr (INRBatch): INR
        dl_args (dict): other parameters

    Returns:
        torch.Tensor: coordinates to sample
    """    
    if dl_args['sample type'] == 'grid':
        coords = _generate_sample_points(inr, method=dl_args['sample type'], dims=dl_args['image shape'])
    else:
        coords = _generate_sample_points(inr, method=dl_args['sample type'], sample_size=dl_args["sample points"])
    return coords

def _generate_sample_points(inr: INRBatch, method: str | None =None,
        sample_size: int | None = None,
        dims: tuple | None =None, ordering: str='c2f') -> PointSet:
    """Generates sample points for integrating along the INR

    Args:
        inr (INRBatch): INR to sample
        method (Optional[str], optional): _description_. Defaults to None.
        sample_size (Optional[int], optional): _description_. Defaults to None.
        dims (Optional[tuple], optional): _description_. Defaults to None.
        ordering (str, optional): _description_. Defaults to 'c2f'.

    Returns:
        torch.Tensor: coordinates to sample
    """
    if method is None:
        method = inr.sample_mode

    if method == "grid":
        if dims is None:
            raise ValueError("declare dims or turn off grid mode")
        return util.meshgrid_coords(*dims, c2f=(ordering=='c2f'))

    elif method == "shrunk":
        coords = generate_quasirandom_sequence(d=inr.input_dims, n=sample_size,
            bbox=(*inr.domain, *inr.domain), scramble=(method=='rqmc'))
        return coords * coords.abs()

    elif method in ("qmc", 'rqmc'):
        return generate_quasirandom_sequence(d=inr.input_dims, n=sample_size,
            bbox=(*inr.domain, *inr.domain), scramble=(method=='rqmc'))

    else:
        raise NotImplementedError("invalid method: "+method)


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
    coords = generate_quasirandom_sequence(d=2, n=int(sample_size/fraction * 1.2),
                bbox=(eps,H-eps,eps,W-eps), scramble=True)
    coo = torch.floor(coords).long()
    bools = mask[coo[:,0], coo[:,1]]
    coord_subset = coords[bools]
    if coord_subset.size(0) < sample_size:
        return generate_masked_sample_points(mask, int(sample_size*1.5))
    return coord_subset[:sample_size]


def generate_quasirandom_sequence(n:int, d:int=2, bbox:tuple=(-1,1,-1,1), scramble:bool=False,
        like=None, dtype=torch.float, device="cuda") -> PointSet:
    """Generates a low discrepancy point set.

    Args:
        n (int): number of points to generate.
        d (int, optional): number of dimensions. Defaults to 2.
        bbox (tuple, optional): edge of domain. Defaults to (-1,1,-1,1).
        scramble (bool, optional): randomized QMC. Defaults to False.
        like (_type_, optional): _description_. Defaults to None.
        dtype (_type_, optional): _description_. Defaults to torch.float.
        device (str, optional): _description_. Defaults to "cuda".

    Returns:
        PointSet (n,d): coordinates
    """
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
    if bbox is not None:
        if d != 2:
            raise NotImplementedError('2D only')
        # bbox has form (x1,x2, y1,y2) in 2D
        out[:,0] = out[:,0] * (bbox[1]-bbox[0]) + bbox[0]
        out[:,1] = out[:,1] * (bbox[3]-bbox[2]) + bbox[2]
    return out.as_subclass(PointSet)


# def linspace(domain, steps, c2f=False):
#     standard_order = torch.linspace(*domain, steps=steps)
#     if c2f:
#         indices = [0]
#         factor = 2
#         cur_step = steps//2
#         while cur_step > 0:
#             indices += list(cur_step * np.arange(1,factor,2))
#             cur_step = cur_step//2
#             factor *= 2
#         return standard_order[torch.tensor(indices)]
#     else:
#         return standard_order
