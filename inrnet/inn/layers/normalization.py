from functools import partial
import torch, pdb
nn = torch.nn
F = nn.functional

from inrnet.inn import functional as inrF

def translate_norm(norm):
    layer = ChannelNorm(channels=norm.num_features, affine=norm.affine,
                momentum=0.1, track_running_stats=norm.track_running_stats)
    if norm.affine:
        layer.weight.data = norm.weight.data
        layer.bias.data = norm.bias.data
    if norm.track_running_stats:
        layer.running_mean.data = norm.running_mean.data
        layer.running_var.data = norm.running_var.data
    layer.eps = norm.eps
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
