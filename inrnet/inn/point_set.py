import torch
from typing import Optional
import numpy as np
import itertools

from inrnet.inn.inr import INRBatch
nn=torch.nn

from inrnet.inn import qmc

class PointSet:
    def __init__(self):
        pass
    def estimate_discrepancy():
        return NotImplemented

def generate_sample_points(inr: INRBatch, dl_args: dict) -> torch.Tensor:
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

def _generate_sample_points(inr: INRBatch, method: Optional[str]=None,
        sample_size: Optional[int]=None,
        dims: Optional[tuple]=None, ordering: str='c2f') -> torch.Tensor:
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
        return meshgrid_coords(*dims, c2f=(ordering=='c2f'))

    elif method == "shrunk":
        coords = qmc.generate_quasirandom_sequence(d=inr.input_dims, n=sample_size,
            bbox=(*inr.domain, *inr.domain), scramble=(method=='rqmc'))
        return coords * coords.abs()

    elif method in ("qmc", 'rqmc'):
        return qmc.generate_quasirandom_sequence(d=inr.input_dims, n=sample_size,
            bbox=(*inr.domain, *inr.domain), scramble=(method=='rqmc'))

    else:
        raise NotImplementedError("invalid method: "+method)

def meshgrid(*tensors, indexing='ij') -> torch.Tensor:
    try:
        return torch.meshgrid(*tensors, indexing=indexing)
    except TypeError:
        return torch.meshgrid(*tensors)

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

def meshgrid_coords(*dims, domain=(-1,1), c2f=True, dtype=torch.float, device="cuda"):
    # c2f: coarse-to-fine ordering, puts points along coarser grid-points first
    tensors = [torch.linspace(*domain, steps=d, dtype=dtype, device=device) for d in dims]
    mgrid = torch.stack(meshgrid(*tensors, indexing='ij'), dim=-1)
        
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
        return flat_grid[indices]

    else:
        return mgrid.reshape(-1, len(dims))
