import torch
nn = torch.nn
F = nn.functional

from inrnet import inn

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, squeeze_ch):
        super().__init__()
        self.se = nn.Sequential(
            inn.GlobalAveragePooling(),
            nn.Linear(channels, squeeze_ch),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze_ch, channels),
            nn.Sigmoid(inplace=True),
        )

    def forward(self, inr):
        weights = self.se(inr)
        return inr * weights

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels=32, radius=.2, spatial_dim=2, dropout=0.,
    		skip, se_ratio, dc_ratio=0.2):
        super().__init__()
        self.layers = [
            conv_bn_act(C, C, radius=radius, input_dims=spatial_dim, groups=C, dropout=dropout),
        	SqueezeExcitation(mid_, int(in_ * se_ratio)) if se_ratio > 0 else nn.Identity(),
        ]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inr):
        expand = self.expand_conv(inputs)
        x = self.depth_wise_conv(expand)
        x = self.se(x)
        x = self.project_conv(x)
        if self.skip:
            x = x + inputs
        return x