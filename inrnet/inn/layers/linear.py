import torch
nn = torch.nn
F = nn.functional
from functools import partial
he_init = nn.init.kaiming_normal_

from inrnet.inn import functional as inrF, polynomials

class ChannelMixer(nn.Module):
    def __init__(self, in_channels, out_channels, normalized=False, bias=True):
        super().__init__()
        self.W_ij = nn.Parameter(torch.empty(in_channels, out_channels))
        he_init(self.W_ij, mode='fan_out', nonlinearity='relu')
        if bias:
            self.b_j = nn.Parameter(torch.zeros(out_channels))
        self.normalized = normalized

    def forward(self, inr):
        if self.normalized:
            out = inr.matmul(torch.softmax(self.W_ij, dim=-1))
        else:
            out = inr.matmul(self.W_ij)
        if hasattr(self, "b_j"):
            return out + self.b_j
        else:
            return out

class AdaptiveChannelMixer(nn.Module):
    def __init__(self, in_channels, out_channels, input_dims=2, normalized=False, bias=True):
        super().__init__()
        self.m_ij = polynomials.LegendreFilter(in_channels, out_channels, input_dims=input_dims).cuda()
        if bias:
            self.b_j = nn.Parameter(torch.zeros(out_channels))
        self.normalized = normalized

    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.integrator = partial(inrF.inner_product, inr=new_inr, layer=self)
        return new_inr
