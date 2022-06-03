import torch, pdb, math
import numpy as np
from functools import partial
nn = torch.nn
F = nn.functional

from inrnet import inn, util
from inrnet.inn import qmc, functional as inrF

def produce_inr(values, **kwargs):
    # values - (B,C,H,W)
    inrs = inn.INRBatch(channels=values.size(1), **kwargs)
    inrs.evaluator = BilinearINR(values=values)
    return inrs

class BilinearINR:
    def __init__(self, values):
        self.values = values
    def __call__(self, coords, eps=1e-6):
        self.sampled_coords = coords
        values = self.values
        B,C,H,W = values.shape
        Tx = torch.linspace(-1-eps,1+eps, steps=H, device=coords.device)
        Ty = torch.linspace(-1-eps,1+eps, steps=W, device=coords.device)
        x_spacing = Tx[1] - Tx[0]
        y_spacing = Ty[1] - Ty[0]

        X = coords[:,0].unsqueeze(1)
        Y = coords[:,1].unsqueeze(1)
        v, kx = (Tx<=X).min(dim=-1)
        if v.max() == True:
            print('out of bounds')
            pdb.set_trace()
        v, ky = (Ty<=Y).min(dim=-1)
        if v.max() == True:
            print('out of bounds')
            pdb.set_trace()

        x_diffs_r = (Tx[kx] - X.squeeze()) / x_spacing
        x_diffs_l = 1-x_diffs_r
        y_diffs_r = (Ty[ky] - Y.squeeze()) / y_spacing
        y_diffs_l = 1-y_diffs_r

        interp_vals = values[:,:,kx,ky]*x_diffs_l*y_diffs_l + \
            values[:,:,kx-1,ky]*x_diffs_r*y_diffs_l + \
            values[:,:,kx,ky-1]*x_diffs_l*y_diffs_r + \
            values[:,:,kx-1,ky-1]*x_diffs_r*y_diffs_r
        return interp_vals.transpose(1,2) #(B,N,C)

class GlobalAvgPoolSequence(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    def forward(self, inr):
        vvf = inr.create_derived_inr()
        vvf.integrator = qmc.Integrator(GAPseq, 'GlobalPoolSeq', layer=self, inr=inr)
        return vvf

def GAPseq(values, layer, inr):
    if inr.training:
        layer.train()
    else:
        layer.eval()
    return layer.layers(values.mean(1).float())
    
class AdaptiveAvgPoolSequence(nn.Module):
    def __init__(self, output_size, layers, extrema=((-1,1),(-1,1))):
        super().__init__()
        self.output_size = output_size
        self.layers = layers
        self.extrema = extrema
    def forward(self, inr):
        vvf = inr.create_derived_inr()
        vvf.integrator = qmc.Integrator(AAPseq, 'AdaptivePoolSeq', layer=self, inr=inr)
        return vvf

def AAPseq(values, layer, inr, eps=1e-6):
    coords = inr.sampled_coords
    if inr.training:
        layer.train()
    else:
        layer.eval()
    h,w = layer.output_size
    if layer.extrema is None:
        layer.extrema = ((coords[:,0].min()-1e-3, coords[:,0].max()+1e-3),
            (coords[:,1].min()-1e-3, coords[:,1].max()+1e-3))

    Tx = torch.linspace(layer.extrema[0][0]-eps, layer.extrema[0][1]+eps, steps=h+1, device=coords.device)
    Ty = torch.linspace(layer.extrema[1][0]-eps, layer.extrema[1][1]+eps, steps=w+1, device=coords.device)

    X = coords[:,0].unsqueeze(1)
    Y = coords[:,1].unsqueeze(1)

    v, kx = (Tx<=X).min(dim=-1)
    if not v.max() == False:
        layer.extrema = ((coords[:,0].min()-1e-3, coords[:,0].max()+1e-3),
            (coords[:,1].min()-1e-3, coords[:,1].max()+1e-3))
        return AAPseq(values, layer, inr)

    v, ky = (Ty<=Y).min(dim=-1)
    if not v.max() == False:
        layer.extrema = ((coords[:,0].min()-1e-3, coords[:,0].max()+1e-3),
            (coords[:,1].min()-1e-3, coords[:,1].max()+1e-3))
        return AAPseq(values, layer, inr)
        
    bins = kx-1 + (ky-1)*h
    out = torch.cat([values[:,bins==b].mean(1) for b in range(h*w)], dim=1)
    return layer.layers(out)
    