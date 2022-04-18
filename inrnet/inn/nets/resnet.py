import torch
nn = torch.nn
F = nn.functional

from inrnet import inn

class EfficientNetB0(nn.Module):
    def __init__(self, in_channels, out_dims, min_channels=16, radius=.2, spatial_dim=2, steerable=False, dropout=0.):
        super().__init__()
        C = min_channels
        self.layers = [
            inn.Conv(in_channels, C, radius=radius, input_dims=spatial_dim, dropout=dropout, steerable=steerable),
            inn.BatchNorm(C),
            inn.ReLU(),
            inn.Conv(C, out_channels),
            inn.GlobalAvgPool(),
        ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inr):
        return self.layers(inr)


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=16, radius=.2, dropout=0.):
        super().__init__()
        C = mid_channels
        self.first = inn.ChannelMixer(in_channels, C)
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

