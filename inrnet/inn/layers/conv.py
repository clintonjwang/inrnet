import torch, pdb
from functools import partial
nn = torch.nn
F = nn.functional

from inrnet.inn import functional as inrF, polynomials

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, radius, stride=False, p_norm="inf",
            input_dims=2, N_bins=16, groups=1, bias=False,
            parameterization="polynomial", padding_mode="cutoff",
            order=3, dropout=0.):
        super().__init__()
        self.radius = radius
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_dims = input_dims
        self.dropout = dropout
        self.N_bins = N_bins
        self.stride = stride
        self.bias = bias
        if p_norm == "inf":
            p_norm = torch.inf
        self.norm = partial(torch.linalg.norm, ord=p_norm, dim=-1)

        if input_dims == 2 and parameterization == "polynomial":
            if p_norm == 2:
                Kernel = polynomials.ZernikeKernel
            elif p_norm == torch.inf:
                Kernel = polynomials.LegendreFilter

            if groups == out_channels and groups == in_channels:
                self.K = Kernel(in_channels=1,
                    out_channels=out_channels, radius=radius, order=order).cuda()
                raise NotImplementedError("TODO: conv groups")
            else:
                self.K = Kernel(in_channels=in_channels, out_channels=out_channels,
                    radius=radius, order=order).cuda()
        else:
            if parameterization == "polynomial":
                raise NotImplementedError("TODO: 3D polynomial basis")

            self.K = nn.Sequential(nn.Linear(input_dims,6), nn.ReLU(inplace=True),
                nn.Linear(6,in_channels*out_channels), Reshape(in_channels,out_channels))
            if groups != 1:
                raise NotImplementedError("TODO: conv groups")
                
        if p_norm not in [2, torch.inf]:
            raise NotImplementedError(f"unsupported norm {p_norm}")
        if parameterization not in ["polynomial", "mlp"]:
            raise NotImplementedError(f"unsupported parameterization {parameterization}")
        if padding_mode not in ["cutoff"]:#, "zeros", "shrink domain", "evaluate"]:
            # cutoff: at each point, evaluate the integral under B intersect I
            # zeros: let the INR be 0 outside I
            # shrink domain: only evaluate points whose ball is contained in I
            # evaluate: sample points outside I
            raise NotImplementedError("TODO: padding modes")
        if bias:
            self.b_j = nn.Parameter(torch.zeros(out_channels))

    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.integrator = partial(inrF.conv, inr=new_inr, layer=self)
        new_inr.channels = self.out_channels
        return new_inr
        
class Reshape(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.reshape(-1, *self.dims)
