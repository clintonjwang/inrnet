import math, torch, pdb
import numpy as np
nn=torch.nn

from inrnet.inn import qmc

class PointSet:
    def __init__(self):
        pass
    def estimate_discrepancy():
        return NotImplemented


def generate_sample_points(inr, dl_args):
    if dl_args['sample type'] == 'grid':
        coords = _generate_sample_points(inr, method=dl_args['sample type'], dims=dl_args['image shape'])
    else:
        coords = _generate_sample_points(inr, method=dl_args['sample type'], sample_size=dl_args["sample points"])
    return coords

def _generate_sample_points(inr, method=None, sample_size=None, dims=None, ordering='c2f'):
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
