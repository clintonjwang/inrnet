import torch
nn = torch.nn
F = nn.functional

from inrnet import inn

def conv_bn_act(in_, out_=32, radius=.2, input_dims=2, **kwargs):
    return nn.Sequential(inn.Conv(in_, out_, radius=radius,
        	input_dims=input_dims, **kwargs),
        inn.BatchNorm(out_),
        inn.SiLU())
