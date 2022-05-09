import torch, pdb, math
import numpy as np
from functools import partial
nn = torch.nn
F = nn.functional

from inrnet import inn, util
from inrnet.inn import functional as inrF
from scipy.interpolate import RectBivariateSpline as Spline2D

def produce_inr(values, **kwargs):
    # values - (B,C,H,W)
    inrs = inn.INRBatch(channels=values.size(1), **kwargs)
    inrs.evaluator = partial(interpolate_values, values=values)
    return inrs

def interpolate_values(self, coords, values):
    B,C,H,W = values.shape
    Tx = torch.linspace(-1,1, steps=H)
    Ty = torch.linspace(-1,1, steps=W)
    x_spacing = Tx[1] - Tx[0]
    y_spacing = Ty[1] - Ty[0]

    X = coords[:,0].unsqueeze(1)
    Y = coords[:,1].unsqueeze(1)
    values, kx = (Tx<=X).min(dim=-1)
    assert values.max() == False
    values, ky = (Ty<=Y).min(dim=-1)
    assert values.max() == False

    x_diffs_r = (Tx[kx] - X.squeeze()) / x_spacing
    x_diffs_l = 1-x_diffs_r
    y_diffs_r = (Ty[ky] - Y.squeeze()) / y_spacing
    y_diffs_l = 1-y_diffs_r

    interp_vals = values[:,:,kx,ky]*x_diffs_l*y_diffs_l + \
        values[:,:,kx-1,ky]*x_diffs_r*y_diffs_l + \
        values[:,:,kx,ky-1]*x_diffs_l*y_diffs_r + \
        values[:,:,kx-1,ky-1]*x_diffs_r*y_diffs_r
    return interp_vals
