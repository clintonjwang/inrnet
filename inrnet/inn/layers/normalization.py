from functools import partial
import torch
nn = torch.nn
F = nn.functional

from inrnet.inn import functional as inrF

def translate_bn2d(bn):
    layer = inn.ChannelNorm(channels=bn.num_features, affine=True,
                momentum=0.1, track_running_stats=True)
    layer.weight.data = bn.weight.data
    layer.bias.data = bn.bias.data
    layer.running_mean.data = bn.running_mean.data
    layer.running_std.data = bn.running_var.data
    return layer

class ChannelNorm(nn.Module):
    def __init__(self, channels=None, affine=True, momentum=0.1,
            track_running_stats=True, device=None, dtype=torch.float):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.affine = affine
        self.momentum = momentum
        if affine:
            self.learned_mean = nn.Parameter(torch.zeros(channels, **factory_kwargs))
            self.learned_std = nn.Parameter(torch.ones(channels, **factory_kwargs))
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(channels, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(channels, **factory_kwargs))

    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.integrator = partial(inrF.normalize, layer=self)
        return new_inr
