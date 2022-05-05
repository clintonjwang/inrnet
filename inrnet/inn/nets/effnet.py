
import torch
nn = torch.nn
F = nn.functional

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


class Conv4(nn.Module):
    def __init__(self, in_channels, out_channels, min_channels=16, spatial_dim=2, **kwargs):
        super().__init__()
        C = min_channels
        conv_kwargs = {"input_dims":spatial_dim, **kwargs}
        self.layers = [
            inn.blocks.conv_norm_act(in_channels, C, radius=.1, **conv_kwargs),
            inn.blocks.ResBlock(C, radius=.2, stride=.1, **conv_kwargs),
            inn.blocks.conv_norm_act(C, C*2, radius=.3, **conv_kwargs),
            inn.blocks.ResBlock(C*2, radius=.4, stride=.2, **conv_kwargs),
            inn.MaxPool(radius=.3),
            inn.blocks.ResBlock(C*2, radius=.6, stride=.4, **conv_kwargs),
            inn.GlobalAvgPool(),
            nn.Linear(C*2, out_channels),
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
