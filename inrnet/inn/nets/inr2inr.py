import torch
nn = torch.nn
F = nn.functional

from inrnet import inn

class ConvCM(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=16, spatial_dim=2, dropout=0.):
        super().__init__()
        C = mid_channels
        self.layers = [
            inn.blocks.conv_bn_act(in_channels, C, radius=.2, input_dims=spatial_dim,
                dropout=dropout),
            inn.AdaptiveChannelMixer(C, out_channels, input_dims=spatial_dim),
        ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inr):
        return self.layers(inr)

class Conv4(nn.Module):
    def __init__(self, in_channels, out_channels, min_channels=16, dropout=0., spatial_dim=2):
        super().__init__()
        C = min_channels
        self.first = inn.AdaptiveChannelMixer(in_channels, C)
        self.layers = [
            inn.MaxPool(radius=.2, stride=.1),
            inn.AdaptiveChannelMixer(C, C*2),
            inn.MaxPool(radius=.4, stride=.2),
            inn.AdaptiveChannelMixer(C*2, C),
        ]
        self.layers = nn.Sequential(*self.layers)
        self.last = inn.AdaptiveChannelMixer(C, out_channels)

    def forward(self, inr):
        z = self.first(inr)
        z = z + self.layers(z)
        return self.last(z)


class Conv4(nn.Module):
    def __init__(self, in_channels, out_channels, min_channels=16, dropout=0., spatial_dim=2):
        super().__init__()
        C = min_channels
        self.first = inn.Conv(in_channels, C, radius=.15, dropout=dropout)
        self.layers = [
            inn.blocks.conv_bn_act(C, C, radius=.2, stride=.1, input_dims=spatial_dim, dropout=dropout),
            inn.ChannelMixer(C, C),
            inn.blocks.conv_bn_act(C, C, radius=.3, stride=.2, input_dims=spatial_dim, dropout=dropout),
            inn.blocks.conv_bn_act(C, C, radius=.4, input_dims=spatial_dim, dropout=dropout),
        ]
        self.layers = nn.Sequential(*self.layers)
        self.last = inn.ChannelMixer(C, out_channels)

    def forward(self, inr):
        z = self.first(inr)
        z = z + self.layers(z)
        return self.last(z)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, min_channels=16):
        super().__init__()
        C = min_channels
        self.pool = inn.MaxPool(radius=radius, p_norm=p_norm)
        self.layers = [
            inn.BatchNorm(C),
            inn.ReLU(),
            inn.ChannelMixer(C, C),
            inn.MaxPool(radius=radius),
        ]
        self.layers = nn.Sequential(*self.layers)
        self.last = inn.ChannelMixer(C, out_channels)

    def forward(self, inr):
        z = self.first(inr)
        z = z + self.layers(z)
        return self.last(z)


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels, min_channels=16):
        super().__init__()
        C = min_channels
        self.pool = inn.MaxPool(radius=radius, p_norm=p_norm)
        self.layers = [
            inn.BatchNorm(C),
            inn.ReLU(),
            inn.ChannelMixer(C, C),
            inn.MaxPool(radius=radius),
        ]
        self.layers = nn.Sequential(*self.layers)
        self.last = inn.ChannelMixer(C, out_channels)

    def forward(self, inr):
        z = self.first(inr)
        z = z + self.layers(z)
        return self.last(z)
