import torch
nn = torch.nn
F = nn.functional
from functools import partial
he_init = nn.init.kaiming_normal_

from inrnet.inn import functional as inrF, polynomials

def translate_conv1x1(conv):
    bias = conv.bias is not None
    layer = ChannelMixer(in_channels=conv.weight.size(1), out_channels=conv.weight.size(0), bias=bias)
    layer.weight.data = conv.weight.data.squeeze(-1).squeeze(-1).T
    if bias:
        layer.bias.data = conv.bias.data
    return layer

class ChannelMixer(nn.Module):
    def __init__(self, in_channels, out_channels, normalized=False, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        he_init(self.weight, mode='fan_out', nonlinearity='relu')
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        self.normalized = normalized

    def __repr__(self):
        return f"""ChannelMixer(in_channels={self.in_channels}, 
            out_channels={self.out_channels}, bias={hasattr(self, 'bias')}, 
            normalized={self.normalized})"""

    def forward(self, inr):
        if self.normalized:
            inr.matmul(torch.softmax(self.weight, dim=-1), inplace=True)
        else:
            inr.matmul(self.weight, inplace=True)
        if hasattr(self, "bias"):
            inr += self.bias
        return inr

class AdaptiveChannelMixer(nn.Module):
    def __init__(self, in_channels, out_channels, input_dims=2, normalized=False, bias=True):
        super().__init__()
        self.m_ij = polynomials.LegendreFilter(in_channels, out_channels, input_dims=input_dims).cuda()
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        self.normalized = normalized

    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.integrator = partial(inrF.inner_product, inr=new_inr, layer=self)
        return new_inr
