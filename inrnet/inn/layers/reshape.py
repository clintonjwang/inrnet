"""INR -> vector or vector -> INR Layer"""
import pdb
import torch

from inrnet.inn.inr import DiscretizedINR
nn = torch.nn
F = nn.functional

from inrnet import inn

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
    def forward(self, inr: DiscretizedINR) -> torch.Tensor:
        if inr.training:
            self.train()
        else:
            self.eval()

        return self.layers(inr.values.mean(1).float())

    
class AdaptiveAvgPoolSequence(nn.Module):
    def __init__(self, output_size, layers, extrema=((-1,1),(-1,1))):
        super().__init__()
        self.output_size = output_size
        self.layers = layers
        self.extrema = extrema
        
    def forward(self, inr: DiscretizedINR, eps=1e-6) -> torch.Tensor:
        if inr.training:
            self.train()
        else:
            self.eval()

        coords = inr.coords
        h,w = self.output_size
        if self.extrema is None:
            self.extrema = ((coords[:,0].min()-1e-3, coords[:,0].max()+1e-3),
                (coords[:,1].min()-1e-3, coords[:,1].max()+1e-3))

        Tx = torch.linspace(self.extrema[0][0]-eps, self.extrema[0][1]+eps, steps=h+1, device=coords.device)
        Ty = torch.linspace(self.extrema[1][0]-eps, self.extrema[1][1]+eps, steps=w+1, device=coords.device)

        X = coords[:,0].unsqueeze(1)
        Y = coords[:,1].unsqueeze(1)

        v, kx = (Tx<=X).min(dim=-1)
        if not v.max() == False:
            self.extrema = ((coords[:,0].min()-1e-3, coords[:,0].max()+1e-3),
                (coords[:,1].min()-1e-3, coords[:,1].max()+1e-3))
            return self.forward(inr)

        v, ky = (Ty<=Y).min(dim=-1)
        if not v.max() == False:
            self.extrema = ((coords[:,0].min()-1e-3, coords[:,0].max()+1e-3),
                (coords[:,1].min()-1e-3, coords[:,1].max()+1e-3))
            return self.forward(inr)
            
        bins = kx-1 + (ky-1)*h
        out = torch.cat([inr.values[:,bins==b].mean(1) for b in range(h*w)], dim=1)
        return self.layers(out)
        