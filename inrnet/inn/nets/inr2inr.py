import torch
nn = torch.nn
F = nn.functional

from inrnet import inn

class ISeg4(nn.Module):
    def __init__(self, in_channels, out_channels, C=16,
            final_activation=None, input_dims=2, **kwargs):
        super().__init__()
        self.layers = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.05,.05), input_dims=input_dims,
                **kwargs),
            inn.blocks.conv_norm_act(C, C, kernel_size=(.05,.05), input_dims=input_dims,
                **kwargs),
            inn.blocks.conv_norm_act(C, C, kernel_size=(.1,.1), input_dims=input_dims,
                **kwargs),
            inn.ChannelMixer(C, out_channels, bias=True),
        ]
        if final_activation is not None:
            self.layers.append(inn.get_activation_layer(final_activation))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inr):
        return self.layers(inr)
        

class ISeg3(nn.Module):
    def __init__(self, in_channels, out_channels, C=16,
            final_activation=None, input_dims=2, **kwargs):
        super().__init__()
        self.layers = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.05,.05), input_dims=input_dims,
                **kwargs),
            inn.blocks.conv_norm_act(C, C, kernel_size=(.05,.05), input_dims=input_dims,
                **kwargs),
            inn.ChannelMixer(C, out_channels, bias=True),
        ]
        if final_activation is not None:
            self.layers.append(inn.get_activation_layer(final_activation))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inr):
        return self.layers(inr)
        

class ISeg5(nn.Module):
    def __init__(self, in_channels, out_channels, C=16, **kwargs):
        super().__init__()
        self.first = nn.Sequential(
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.05,.05), **kwargs),
            inn.blocks.conv_norm_act(C, C, kernel_size=(.05,.05), **kwargs))
        layers = [
            inn.blocks.conv_norm_act(C, C, kernel_size=(.1,.1), **kwargs),
            inn.blocks.conv_norm_act(C, C, kernel_size=(.1,.1), **kwargs),
        ]
        self.layers = nn.Sequential(*layers)
        self.last = nn.Sequential(
            inn.ChannelMixer(C, out_channels))

    def forward(self, inr):
        inr = self.first(inr)
        inr = inr + self.layers(inr.create_derived_inr())
        return self.last(inr)

