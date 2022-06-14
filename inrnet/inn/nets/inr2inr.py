import torch

from inrnet.inn.inr import DiscretizedINR
nn = torch.nn
F = nn.functional

from inrnet import inn
from inrnet.inn.nets.inrnet import INRNet

class ISeg3(INRNet):
    def __init__(self, in_channels, out_channels, sampler=None, C=16,
            final_activation=None, **kwargs):
        layers = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.025,.05),
                **kwargs),
            inn.PositionalEncoding(N=C//4),
            inn.blocks.conv_norm_act(C, C*2, kernel_size=(.075,.15),
                **kwargs),
            inn.ChannelMixer(C*2, out_channels, bias=True),
        ]
        if final_activation is not None:
            layers.append(inn.get_activation_layer(final_activation))
        layers = nn.Sequential(*layers)
        super().__init__(sampler=sampler, layers=layers)


class ISeg5(INRNet):
    def __init__(self, in_channels, out_channels, sampler=None, C=16, **kwargs):
        super().__init__(sampler=sampler)
        self.first = nn.Sequential(
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.03,.03), **kwargs),
            inn.PositionalEncoding(N=C//4))
        layers = [
            inn.blocks.conv_norm_act(C, C, kernel_size=(.06,.06), down_ratio=.5, **kwargs),
            inn.blocks.conv_norm_act(C, C, kernel_size=(.1,.1), down_ratio=.5, **kwargs),
            inn.Upsample(4),
            inn.blocks.conv_norm_act(C, C, kernel_size=(.06,.06), **kwargs),
        ]
        self.layers = nn.Sequential(*layers)
        self.last = nn.Sequential(
            inn.ChannelMixer(C, out_channels))

    def _forward(self, inr: DiscretizedINR) -> DiscretizedINR|torch.Tensor:
        inr = self.first(inr)
        inr = inr + self.layers(inr.create_derived_inr())
        return self.last(inr)
