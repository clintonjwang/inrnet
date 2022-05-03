import torch, pdb, math
import numpy as np
from functools import partial
nn = torch.nn
F = nn.functional

from inrnet import util

def translate_upsample(layer, input_shape, extrema):
    h,w = input_shape
    extrema_dists = extrema[0][1] - extrema[0][0], extrema[1][1] - extrema[1][0]
    spacing = extrema_dists[0]/2 / (h-1), extrema_dists[1]/2 / (w-1)
    output_shape = input_shape[0]*2, input_shape[0]*2
    extrema = ((extrema[0][0], extrema[0][1]+spacing[0]),
            (extrema[1][0], extrema[1][1]+spacing[1]))
    if layer.scale_factor in [2,(2,2)]:
        return Upsample(4, mode=layer.mode, spacing=spacing, shift=(spacing[0]/2, spacing[1]/2)), output_shape, extrema
    raise NotImplementedError


def upsample_nn(values, inr, layer):
    coords = inr.sampled_coords
    down_size = coords.size(0)
    if inr.grid_mode:
        new_coords = torch.cat((
            torch.stack((coords[:,0]+layer.spacing[0], coords[:,1]+layer.spacing[1]), dim=1),
            torch.stack((coords[:,0], coords[:,1]+layer.spacing[1]), dim=1),
            torch.stack((coords[:,0]+layer.spacing[0], coords[:,1]), dim=1)
        ), dim=0)
    else:
        new_coords = util.generate_quasirandom_sequence(n=down_size*layer.scale, d=coords.size(1), like=coords)[down_size:]
    
    Diffs = ((new_coords - layer.shift).unsqueeze(1) - coords.unsqueeze(0)).norm(dim=-1)
    new_values = values[Diffs.min(dim=1).indices]
    inr.sampled_coords = torch.cat((coords, new_coords), dim=0)
    return torch.cat((values, new_values), dim=0)


def upsample_conv(values, inr, layer):
    coords = inr.sampled_coords
    down_size = coords.size(0)
    if inr.grid_mode:
        new_coords = torch.cat((
            torch.stack((coords[:,0]+layer.spacing[0], coords[:,1]+layer.spacing[1]), dim=1),
            torch.stack((coords[:,0], coords[:,1]+layer.spacing[1]), dim=1),
            torch.stack((coords[:,0]+layer.spacing[0], coords[:,1]), dim=1)
        ), dim=0)
    else:
        new_coords = util.generate_quasirandom_sequence(n=down_size*layer.scale, d=coords.size(1), like=coords)[down_size:]
    
    Diffs = ((new_coords - layer.shift).unsqueeze(1) - coords.unsqueeze(0)).norm(dim=-1)
    new_values = values[Diffs.min(dim=1).indices]
    inr.sampled_coords = torch.cat((coords, new_coords), dim=0)
    return torch.cat((values, new_values), dim=0)


class Upsample(nn.Module):
    def __init__(self, scale, mode='nearest', kernel=None,
            spacing=None, shift=(0,0), dtype=torch.float):
        super().__init__()
        self.scale = scale
        self.mode = mode
        self.spacing = spacing
        if self.mode != 'nearest':
            if kernel is not None:
                self.kernel = kernel
            else:
                raise NotImplementedError()
                self.kernel = get_kernel_for_mode(mode, scale)
        self.register_buffer('shift', torch.tensor(shift, dtype=dtype))

    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        if self.mode == 'nearest':
            new_inr.set_integrator(upsample_nn, 'AvgPool', layer=self)
        else:
            new_inr.set_integrator(upsample_conv, 'AvgPool', layer=self)
        return new_inr
