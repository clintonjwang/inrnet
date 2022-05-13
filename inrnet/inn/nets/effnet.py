from inrnet import inn
from inrnet.inn import functional as inrF

import torch
nn = torch.nn
F = nn.functional

class SimpleConv(nn.Module):
    def __init__(self, in_channels, out_dims, C=32, **kwargs):
        super().__init__()
        #nn.Linear(C*2, 64), nn.LeakyReLU(inplace=True), 
        out_layers = nn.Sequential(nn.Linear(64, out_dims))
        for l in out_layers:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
                nn.init.zeros_(l.bias)
        self.layers = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.05,.05), **kwargs),
            inn.MaxPool(2, stride=2),
            inn.blocks.conv_norm_act(C, C*2, kernel_size=(.1,.1), **kwargs),
            # inn.blocks.conv_norm_act(C*2, C*2, kernel_size=(.1,.1), **kwargs),
            inn.GlobalAvgPoolSequence(out_layers),
        ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inr):
        return self.layers(inr)


class SimpleConv2(nn.Module):
    def __init__(self, in_channels, out_dims, C=24, **kwargs):
        super().__init__()
        out_layers = nn.Sequential(nn.Linear(C*2, 64), nn.LeakyReLU(inplace=True), nn.Linear(64, out_dims))
        for l in out_layers:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
                nn.init.zeros_(l.bias)
        self.layers = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.05,.05), stride=2, **kwargs),
            inn.blocks.ResConv(C, kernel_size=(.1,.1), **kwargs),
            inn.MaxPool((.08,.08)),
            inn.blocks.conv_norm_act(C, C*2, kernel_size=(.15,.15), **kwargs),
            inn.blocks.ResConv(C*2, kernel_size=(.2,.2), **kwargs),
            inn.GlobalAvgPoolSequence(out_layers),
        ]
        self.layers = nn.Sequential(*self.layers)
    def forward(self, inr):
        return self.layers(inr)


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=16, dropout=0.):
        super().__init__()
        C = mid_channels
        self.first = inn.ChannelMixer(in_channels, C)
        self.layers = [
            inn.ResBlock(C),
            inn.ResBlock(),
            inn.ResBlock(C, C),
            inn.MaxPool(radius=.2),
            inn.GlobalAvgPool(),
            nn.Linear(C*2, out_channels),
        ]
        self.layers = nn.Sequential(*self.layers)
        self.last = inn.ChannelMixer(C, out_channels)

    def forward(self, inr):
        z = self.first(inr)
        z = z + self.layers(z)
        return self.last(z)
