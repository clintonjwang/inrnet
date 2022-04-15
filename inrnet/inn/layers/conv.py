import torch
from functools import partial
nn = torch.nn
F = nn.functional
he_init = nn.init.kaiming_normal_

from inrnet.inn import integrate

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, radius, p_norm=2,
            input_dims=2, groups=1, bias=False, padding_mode="cutoff",
            dropout=0.):
        super().__init__()
        self.radius = radius
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_dims = input_dims
        self.dropout = dropout

        self.K = nn.Sequential(nn.Linear(input_dims,2), nn.ReLU(inplace=True),
            nn.Linear(2,in_channels*out_channels))
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
        self.norm = partial(torch.linalg.norm, ord=p_norm, dim=-1)

    def forward(self, inr):
        return inr.conv(self)


class ChannelMixing(nn.Module):
    def __init__(self, in_channels, out_channels, normalized=False):
        super().__init__()
        self.W_ij = nn.Parameter(torch.empty(in_channels, out_channels))
        he_init(self.W_ij, mode='fan_out', nonlinearity='relu')
        self.b_j = nn.Parameter(torch.zeros(out_channels))
        self.normalized = normalized

    def forward(self, inr):
        if self.normalized:
            return inr.matmul(torch.softmax(self.W_ij, dim=1)) + self.b_j
        else:
            return inr.matmul(self.W_ij) + self.b_j

class AdaptiveChannelMixing(nn.Module):
    def __init__(self, in_channels, out_channels, degrees_of_freedom, normalized=False):
        super().__init__()
        self.m = nn.Parameter(torch.empty(in_channels, out_channels, degrees_of_freedom))

    def forward(self, inr):
        raise NotImplementedError("TODO")
        G = lambda dx,dy: (inr(x-dx,y-dy) * K(dx,dy)).reshape(-1, in_channels, out_channels).sum(1)
        return inr.integrate2d(G, -k[0], k[0], -k[1], k[1])

