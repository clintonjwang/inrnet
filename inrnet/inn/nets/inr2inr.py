import torch
nn = torch.nn
F = nn.functional

from inrnet import inn


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, min_channels=16, spatial_dim=2, final_activation=None, **kwargs):
        super().__init__()
        C = min_channels
        conv_kwargs = {"input_dims":spatial_dim, **kwargs}
        self.first = inn.blocks.conv_norm_act(in_channels, C, radius=.1, **conv_kwargs)
        self.down1 = inn.blocks.conv_norm_act(C, C*2, radius=.2, stride=.1, **conv_kwargs)
        self.down2 = inn.blocks.conv_norm_act(C*2, C*4, radius=.4, stride=.2, **conv_kwargs)
        self.down3 = nn.Sequential(
            inn.blocks.conv_norm_act(C*4, C*8, radius=.7, stride=.4, **conv_kwargs),
            inn.blocks.ResBlock(C*8, radius=.8, **conv_kwargs),
        )
        self.up3 = inn.blocks.conv_norm_act(C*8, C*4, radius=1., **conv_kwargs)
        self.up2 = inn.blocks.conv_norm_act(C*4, C*2, radius=.6, **conv_kwargs)
        self.up1 = inn.blocks.conv_norm_act(C*2, C, radius=.3, **conv_kwargs)
        self.last = [
            inn.blocks.conv_norm_act(C, C, radius=.1, **conv_kwargs),
            inn.ChannelMixer(C, out_channels)
        ]
        if final_activation is not None:
            self.last.append(inn.get_activation_layer(final_activation))
        self.last = nn.Sequential(*self.last)

    def forward(self, inr):
        z1 = self.first(inr)
        z2 = self.down1(z1)
        z3 = self.down2(z2)
        z = self.down3(z3)
        z = self.up3(z) + z3
        z = self.up2(z) + z2
        z = self.up1(z) + z1
        return self.last(z)


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels, min_channels=16, spatial_dim=2, final_activation=None, **kwargs):
        super().__init__()
        C = min_channels
        conv_kwargs = {"input_dims":spatial_dim, **kwargs}
        self.first = inn.blocks.conv_norm_act(in_channels, C, radius=.1, **conv_kwargs)
        self.down1 = nn.Sequential(
            # inn.blocks.ResBlock(C, radius=.2, stride=.1, **conv_kwargs),
            inn.blocks.conv_norm_act(C, C*2, radius=.3, stride=.1, **conv_kwargs),
        )
        self.down2 = nn.Sequential(
            inn.blocks.ResBlock(C*2, radius=.4, stride=.2, **conv_kwargs),
            # inn.blocks.MaxPool(radius=.3),
            inn.blocks.conv_norm_act(C*2, C*4, radius=.5, **conv_kwargs),
        )
        self.down3 = nn.Sequential(
            inn.blocks.conv_norm_act(C*4, C*8, radius=.7, stride=.4, **conv_kwargs),
            inn.blocks.ResBlock(C*8, radius=.8, **conv_kwargs),
        )
        self.up3 = inn.blocks.conv_norm_act(C*8, C*4, radius=1., **conv_kwargs)
        self.up2 = nn.Sequential(
            inn.blocks.conv_norm_act(C*4, C*2, radius=.6, **conv_kwargs),
            # inn.blocks.ResBlock(C*2, radius=.4, **conv_kwargs),
        )
        self.up1 = inn.blocks.conv_norm_act(C*2, C, radius=.2, **conv_kwargs)
        self.to_out1 = nn.Sequential(
            inn.blocks.conv_norm_act(C,C, radius=.1, **conv_kwargs),
            inn.ChannelMixer(C, out_channels)
        )
        self.to_out2 = nn.Sequential(
            inn.blocks.conv_norm_act(C*2,out_channels, radius=.2, **conv_kwargs),
        )
        self.to_out3 = nn.Sequential(
            inn.blocks.conv_norm_act(C*4,out_channels, radius=.4, **conv_kwargs),
        )
        self.to_out4 = nn.Sequential(
            inn.blocks.conv_norm_act(C*8,out_channels, radius=.7, **conv_kwargs),
        )
        # self.to_out2 = inn.ChannelMixer(C*2, out_channels)
        # self.to_out3 = inn.ChannelMixer(C*4, out_channels)
        # self.to_out4 = inn.ChannelMixer(C*8, out_channels)
        if final_activation is not None:
            self.final = inn.get_activation_layer(final_activation)

    def forward(self, inr):
        d1 = self.first(inr)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        u3 = self.up3(d4) + d3
        u2 = self.up2(u3) + d2
        u1 = self.up1(u2) + d1
        out = self.to_out1(u1) + self.to_out2(u2) + self.to_out3(u3) + self.to_out4(d4)
        if hasattr(self, "final"):
            out = self.final(out)
        return out


class ConvCM(nn.Module):
    def __init__(self, in_channels, out_channels, min_channels=16,
            final_activation=None, spatial_dim=2, **kwargs):
        super().__init__()
        C = min_channels
        self.layers = [
            inn.blocks.conv_norm_act(in_channels, out_channels, radius=.01, input_dims=spatial_dim,
                **kwargs),
            # inn.AdaptiveChannelMixer(in_channels, out_channels, input_dims=spatial_dim, bias=True),
        ]
        if final_activation is not None:
            self.layers.append(inn.get_activation_layer(final_activation))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inr):
        return self.layers(inr)
        

class CM4(nn.Module):
    def __init__(self, in_channels, out_channels, min_channels=16, dropout=0., spatial_dim=2):
        super().__init__()
        C = min_channels
        self.first = inn.ChannelMixer(in_channels, C)
        self.layers = [
            inn.MaxPool(radius=.2, stride=.1),
            inn.AdaptiveChannelMixer(C, C*2),
            inn.MaxPool(radius=.4, stride=.2),
            inn.AdaptiveChannelMixer(C*2, C),
        ]
        self.layers = nn.Sequential(*self.layers)
        self.last = nn.Sequential(
            inn.blocks.conv_norm_act(C, C, radius=.1, input_dims=spatial_dim, dropout=dropout),
            inn.ChannelMixer(C, out_channels))

    def forward(self, inr):
        z = self.first(inr)
        z = z + self.layers(z)
        return self.last(z)

