"""Normalization Layer"""
from typing import Optional
import torch
nn = torch.nn
F = nn.functional

from inrnet.inn import functional as inrF

def translate_norm(norm):
    layer = ChannelNorm(channels=norm.num_features, batchnorm=isinstance(norm, nn.modules.batchnorm._BatchNorm),
            affine=norm.affine, momentum=0.1, track_running_stats=norm.track_running_stats)
    if norm.affine:
        layer.weight.data = norm.weight.data
        layer.bias.data = norm.bias.data
    if norm.track_running_stats:
        layer.running_mean.data = norm.running_mean.data
        layer.running_var.data = norm.running_var.data
    layer.eps = norm.eps
    return layer


class ChannelNorm(nn.Module):
    def __init__(self, channels:Optional[int]=None, batchnorm:bool=True,
            affine:bool=True, momentum:float=0.1,
            track_running_stats:bool=True, eps:float=1e-5,
            device=None, dtype=torch.float):
        """Encompasses Batch Norm and Instance Norm

        Args:
            channels (_type_, optional): _description_. Defaults to None.
            batchnorm (bool, optional): _description_. Defaults to True.
            affine (bool, optional): _description_. Defaults to True.
            momentum (float, optional): _description_. Defaults to 0.1.
            track_running_stats (bool, optional): _description_. Defaults to True.
            eps (_type_, optional): _description_. Defaults to 1e-5.
            device (_type_, optional): _description_. Defaults to None.
            dtype (_type_, optional): _description_. Defaults to torch.float.
        """            
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.momentum = momentum
        self.eps = eps
        self.batchnorm = batchnorm
        if affine:
            self.bias = nn.Parameter(torch.zeros(channels, **factory_kwargs))
            self.weight = nn.Parameter(torch.ones(channels, **factory_kwargs))
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(channels, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(channels, **factory_kwargs))

    def __repr__(self):
        return f"ChannelNorm(batch={self.batchnorm}, affine={hasattr(self, 'weight')}, track_running_stats={hasattr(self,'running_mean')})"

    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        if self.batchnorm:
            new_inr.set_integrator(inrF.batch_normalize, 'BatchNorm', layer=self)
        else:
            new_inr.set_integrator(inrF.inst_normalize, 'InstanceNorm', layer=self)
        return new_inr
