from inrnet import inn

import torch
nn = torch.nn
F = nn.functional

class InrCls(nn.Module):
    def __init__(self, in_channels, out_dims, C=64, **kwargs):
        super().__init__()
        out_layers = nn.Sequential(nn.Linear(C*2, 128), nn.ReLU(inplace=True), nn.Linear(128, out_dims))
        for l in out_layers:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
                nn.init.zeros_(l.bias)
        # kwargs.pop('mid_ch', None);
        # kwargs.pop('N_bins', None);
        # kwargs.pop('scale2', None);
        k0 = kwargs.pop('k0', .05)
        k1 = kwargs.pop('k1', .1)
        k2 = kwargs.pop('k2', .2)
        self.layers = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(k0,k0), down_ratio=.25, **kwargs),
            inn.PositionalEncoding(N=C//4),
            inn.blocks.ResConv(C, kernel_size=(k1,k1), **kwargs),
            inn.ChannelMixer(C, C*2),
            inn.MaxPool(kernel_size=(.1,.1), down_ratio=.25),
            inn.blocks.ResConv(C*2, kernel_size=(k2,k2), **kwargs),
            inn.GlobalAvgPoolSequence(out_layers),
        ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inr):
        return self.layers(inr)


class InrCls2(nn.Module):
    def __init__(self, in_channels, out_dims, C=64, **kwargs):
        super().__init__()
        out_layers = nn.Sequential(nn.Linear(C*2, 128), nn.ReLU(inplace=True), nn.Linear(128, out_dims))
        for l in out_layers:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
                nn.init.zeros_(l.bias)
        self.layers = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.05,.05), **kwargs),
            inn.PositionalEncoding(N=C//4),
            inn.blocks.conv_norm_act(C, C*2, kernel_size=(.2,.2), down_ratio=.25, **kwargs),
            inn.GlobalAvgPoolSequence(out_layers),
        ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inr):
        return self.layers(inr)

class AAPCls2(nn.Module):
    def __init__(self, in_channels, out_dims, C=64, **kwargs):
        super().__init__()
        out_layers = nn.Sequential(nn.Linear(C*16, 128), nn.ReLU(inplace=True), nn.Linear(128, out_dims))
        for l in out_layers:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
                nn.init.zeros_(l.bias)
        self.layers = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.05,.05), **kwargs),
            inn.PositionalEncoding(N=C//4),
            inn.blocks.conv_norm_act(C, C, kernel_size=(.2,.2), down_ratio=.25, **kwargs),
            inn.AdaptiveAvgPoolSequence((4,4), out_layers, extrema=None),
        ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inr):
        return self.layers(inr)


class InrCls4(nn.Module):
    def __init__(self, in_channels, out_dims, C=32, **kwargs):
        super().__init__()
        out_layers = nn.Sequential(nn.Linear(C*2, 128), nn.ReLU(inplace=True), nn.Linear(128, out_dims))
        for l in out_layers:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
                nn.init.zeros_(l.bias)
        self.layers = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.05,.05), **kwargs),
            inn.PositionalEncoding(N=C//4),
            inn.blocks.conv_norm_act(C, C*2, kernel_size=(.2,.2), down_ratio=.25, **kwargs),
            inn.blocks.ResConv(C*2, kernel_size=(.3,.3), **kwargs),
            inn.GlobalAvgPoolSequence(out_layers),
        ]
        self.layers = nn.Sequential(*self.layers)
    def forward(self, inr):
        return self.layers(inr)

class AAPInrCls4(nn.Module):
    def __init__(self, in_channels, out_dims, C=32, **kwargs):
        super().__init__()
        out_layers = nn.Sequential(nn.Linear(C*2*16, 128), nn.ReLU(inplace=True), nn.Linear(128, out_dims))
        for l in out_layers:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
                nn.init.zeros_(l.bias)
        self.layers = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.05,.05), **kwargs),
            inn.PositionalEncoding(N=C//4),
            inn.blocks.conv_norm_act(C, C*2, kernel_size=(.2,.2), down_ratio=.25, **kwargs),
            inn.blocks.ResConv(C*2, kernel_size=(.3,.3), **kwargs),
            inn.AdaptiveAvgPoolSequence((4,4), out_layers, extrema=None),
        ]
        self.layers = nn.Sequential(*self.layers)
    def forward(self, inr):
        return self.layers(inr)

class InrClsWide4(nn.Module):
    def __init__(self, in_channels, out_dims, C=32, **kwargs):
        super().__init__()
        out_layers = nn.Sequential(nn.Linear(C*2, 128), nn.ReLU(inplace=True), nn.Linear(128, out_dims))
        for l in out_layers:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
                nn.init.zeros_(l.bias)
        kwargs.pop('mid_ch', None);
        kwargs.pop('N_bins', None);
        self.layers = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.05,.05), **kwargs),
            inn.PositionalEncoding(N=C//4),
            inn.blocks.conv_norm_act(C, C*2, kernel_size=(.3,.3), down_ratio=.25, **kwargs),
            inn.blocks.ResConv(C*2, kernel_size=(.3,.3), **kwargs),
            # inn.blocks.conv_norm_act(C*2, C*2, kernel_size=(.3,.3), down_ratio=.5,
            #     N_bins=256, mid_ch=(16,32), **kwargs),
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
