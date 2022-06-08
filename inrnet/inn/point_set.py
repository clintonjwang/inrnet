"""Point set"""
import torch
from typing import Optional
import numpy as np
import itertools

from inrnet.inn.inr import INRBatch
nn=torch.nn

from inrnet import util
from inrnet.inn import qmc

class PointValues(torch.Tensor):
    def batch_size(self):
        return self.size(0)
    def N(self):
        return self.size(-2)
    def channels(self):
        return self.size(-1)
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    
class PointSet(torch.Tensor):
    def N(self):
        return self.size(-2)
    def dims(self):
        return self.size(-1)
    # def estimate_discrepancy():
    #     return NotImplemented

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
        dims: Optional[tuple]=None, ordering: str='c2f') -> PointSet:
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
        coords = qmc.generate_quasirandom_sequence(d=inr.input_dims, n=sample_size,
            bbox=(*inr.domain, *inr.domain), scramble=(method=='rqmc'))
        return coords * coords.abs()

    elif method in ("qmc", 'rqmc'):
        return qmc.generate_quasirandom_sequence(d=inr.input_dims, n=sample_size,
            bbox=(*inr.domain, *inr.domain), scramble=(method=='rqmc'))

    else:
        raise NotImplementedError("invalid method: "+method)

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
