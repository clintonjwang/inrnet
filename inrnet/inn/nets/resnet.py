import torch
nn = torch.nn
F = nn.functional

from inrnet import inn

class ConvCM(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=16, radius=.2, spatial_dim=2):
        super().__init__()
        C = mid_channels
        self.layers = [
            inn.Conv(in_channels, C, radius=radius, input_dims=spatial_dim),
            #inn.BatchNorm(C),
            inn.InstanceNorm(C),
            inn.ReLU(),
            inn.ChannelMixing(C, out_channels),
        ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inr):
        return self.layers(inr)


class ConvCmConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=16, radius=.2, spatial_dim=2):
        super().__init__()
        C = mid_channels
        self.layers = [
            inn.Conv(in_channels, C, radius=radius, input_dims=spatial_dim),
            inn.BatchNorm(C),
            inn.ReLU(),
            inn.ChannelMixing(C, C),
            inn.BatchNorm(C),
            inn.ReLU(),
            inn.Conv(C, out_channels, radius=radius, input_dims=spatial_dim),
        ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inr):
        return self.layers(inr)


class CmPlCm(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=16, radius=.2, spatial_dim=2):
        super().__init__()
        C = mid_channels
        self.layers = [
            inn.ChannelMixing(in_channels, C),
            inn.BatchNorm(C),
            inn.ReLU(),
            inn.ChannelMixing(C, C),
            inn.MaxPool(radius=radius),
            inn.ChannelMixing(C, out_channels),
        ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inr):
        return self.layers(inr)


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=16):
        super().__init__()
        self.first = inn.ChannelMixing(in_channels, C),
        self.layers = [
            inn.BatchNorm(C),
            inn.ReLU(),
            inn.ChannelMixing(C, C),
            inn.MaxPool(radius=radius),
        ]
        self.layers = nn.Sequential(*self.layers)
        self.last = inn.ChannelMixing(C, out_channels)

    def forward(self, inr):
        z = self.first(inr)
        z = z + self.layers(z)
        return self.last(z)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=16):
        super().__init__()
        self.pool = inn.MaxPool(radius=radius, p_norm=p_norm)
        self.layers = [
            inn.BatchNorm(C),
            inn.ReLU(),
            inn.ChannelMixing(C, C),
            inn.MaxPool(radius=radius),
        ]
        self.layers = nn.Sequential(*self.layers)
        self.last = inn.ChannelMixing(C, out_channels)

    def forward(self, inr):
        z = self.first(inr)
        z = z + self.layers(z)
        return self.last(z)


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=16):
        super().__init__()
        self.pool = inn.MaxPool(radius=radius, p_norm=p_norm)
        self.layers = [
            inn.BatchNorm(C),
            inn.ReLU(),
            inn.ChannelMixing(C, C),
            inn.MaxPool(radius=radius),
        ]
        self.layers = nn.Sequential(*self.layers)
        self.last = inn.ChannelMixing(C, out_channels)

    def forward(self, inr):
        z = self.first(inr)
        z = z + self.layers(z)
        return self.last(z)
