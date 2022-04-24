import torch
nn = torch.nn
F = nn.functional

from inrnet import inn

def conv_norm_act(in_, out_, radius=.2, **kwargs):
    act_layer = inn.get_activation_layer(kwargs.pop("activation", "swish"))
    return nn.Sequential(inn.Conv(in_, out_, radius=radius, **kwargs),
        inn.ChannelNorm(out_),
        act_layer,
    )

class ResBlock(nn.Module):
    def __init__(self, sequential):
        super().__init__()
        self.sequential = sequential
    def forward(self, inr):
        # return inr + self.sequential(inr)
        return inn.ResINR(inr, self.sequential)

class ResConv(nn.Module):
    def __init__(self, C, **kwargs):
        super().__init__()
        stride = kwargs.pop("stride", 0)
        self.layers = nn.Sequential(conv_norm_act(C, C, stride=0, **kwargs),
            conv_norm_act(C, C, stride=stride, **kwargs))
    def forward(self, inr):
        return self.layers(inr) + inr
