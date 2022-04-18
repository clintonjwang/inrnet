import torch, pdb
from functools import partial
nn = torch.nn
F = nn.functional
he_init = nn.init.kaiming_normal_

from inrnet.inn import functional as inrF, polynomials

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, radius, stride=False, p_norm=2,
            input_dims=2, N_bins=16, groups=1, bias=False, padding_mode="cutoff",
            order=3, dropout=0.):
        super().__init__()
        self.radius = radius
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_dims = input_dims
        self.dropout = dropout
        self.N_bins = N_bins
        self.stride = stride
        self.norm = partial(torch.linalg.norm, ord=p_norm, dim=-1)

        if input_dims == 2:
            if groups == out_channels and groups == in_channels:
                self.K = polynomials.ZernikeKernel(in_channels=1,
                    out_channels=out_channels, radius=radius, order=order)
            else:
                self.K = polynomials.ZernikeKernel(in_channels=in_channels, #CircularHarmonics
                    out_channels=out_channels, radius=radius, order=order)
        else:
            self.K = nn.Sequential(nn.Linear(input_dims,2), nn.ReLU(inplace=True),
                nn.Linear(2,in_channels*out_channels), Reshape(in_channels,out_channels))

        if groups != 1:
            raise NotImplementedError("TODO: conv groups")
        if padding_mode not in ["cutoff"]:#, "zeros", "shrink domain", "evaluate"]:
            # cutoff: at each point, evaluate the integral under B intersect I
            # zeros: let the INR be 0 outside I
            # shrink domain: only evaluate points whose ball is contained in I
            # evaluate: sample points outside I
            raise NotImplementedError("TODO: padding modes")
        if bias:
            raise NotImplementedError("TODO: bias")
        if stride:
            raise NotImplementedError("stride")

    def forward(self, inr):
        inr.integrator = partial(inrF.conv, inr=inr, layer=self)
        inr.channels = self.out_channels
        return inr.create_derived_inr()
        
class Reshape(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.reshape(-1, *self.dims)

class ChannelMixer(nn.Module):
    def __init__(self, in_channels, out_channels, normalized=False):
        super().__init__()
        self.W_ij = nn.Parameter(torch.empty(in_channels, out_channels))
        he_init(self.W_ij, mode='fan_out', nonlinearity='relu')
        self.b_j = nn.Parameter(torch.zeros(out_channels))
        self.normalized = normalized

    def forward(self, inr):
        if self.normalized:
            return inr.matmul(torch.softmax(self.W_ij, dim=-1)) + self.b_j
        else:
            return inr.matmul(self.W_ij) + self.b_j


class AdaptiveChannelMixer(nn.Module):
    def __init__(self, in_channels, out_channels, input_dims=2, normalized=False):
        super().__init__()
        self.m_ij = polynomials.LegendreFilter(in_channels, out_channels, input_dims=input_dims).cuda()
        self.b_j = nn.Parameter(torch.zeros(out_channels))
        self.normalized = normalized

    def forward(self, inr):
        inr.integrator = partial(inrF.inner_product, layer=self)
        return inr.create_derived_inr()
