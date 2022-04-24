from functools import partial
import torch, pdb
nn = torch.nn
F = nn.functional

from inrnet.inn import functional as inrF

def translate_bn(bn):
    layer = ChannelNorm(channels=bn.num_features, affine=True,
                momentum=0.1, track_running_stats=True)
    layer.weight.data = bn.weight.data
    layer.bias.data = bn.bias.data
    layer.running_mean.data = bn.running_mean.data
    layer.running_var.data = bn.running_var.data
    layer.eps = bn.eps
    return layer

class ChannelNorm(nn.Module):
    def __init__(self, channels=None, affine=True, momentum=0.1,
            track_running_stats=True, eps=1e-5, device=None, dtype=torch.float):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.momentum = momentum
        self.eps = eps
        if affine:
            self.bias = nn.Parameter(torch.zeros(channels, **factory_kwargs))
            self.weight = nn.Parameter(torch.ones(channels, **factory_kwargs))
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(channels, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(channels, **factory_kwargs))

    def __repr__(self):
        return f"ChannelNorm(affine={hasattr(self, 'weight')}, track_running_stats={hasattr(self,'running_mean')})"

    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.integrator = partial(inrF.normalize, inr=inr, layer=self)
        return new_inr
