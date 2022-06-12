"""Normalization Layer"""
from typing import Optional
import torch

from inrnet.inn.inr import DiscretizedINR, INRBatch
from inrnet.inn.point_set import PointValues
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

    def __str__(self):
        if self.batchnorm:
            return 'BN'
        else:
            return 'LN'
    def __repr__(self):
        return f"ChannelNorm(batch={self.batchnorm}, affine={hasattr(self, 'weight')}, track_running_stats={hasattr(self,'running_mean')})"

    def inst_normalize(self, values, inr: DiscretizedINR) -> DiscretizedINR:
        if hasattr(self, "running_mean") and not (inr.training and self.training):
            mean = self.running_mean
            var = self.running_var
        else:
            mean = values.mean(1, keepdim=True)
            var = values.pow(2).mean(1, keepdim=True) - mean.pow(2)
            if hasattr(self, "running_mean"):
                with torch.no_grad():
                    self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mean.mean()
                    self.running_var = self.momentum * self.running_var + (1-self.momentum) * var.mean()
                mean = self.running_mean
                var = self.running_var

        return self.normalize(values, mean, var)

    def batch_normalize(self, values, inr: DiscretizedINR) -> DiscretizedINR:
        if hasattr(self, "running_mean") and not (inr.training and self.training):
            mean = self.running_mean
            var = self.running_var
        else:
            mean = values.mean(dim=(0,1))
            var = values.pow(2).mean(dim=(0,1)) - mean.pow(2)
            if hasattr(self, "running_mean"):
                with torch.no_grad():
                    self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mean
                    self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
                mean = self.running_mean
                var = self.running_var

        return self.normalize(values, mean, var)

    def normalize(self, values: PointValues, mean: torch.Tensor, var: torch.Tensor):
        if hasattr(self, "weight"):
            return (values - mean)/(var.sqrt() + self.eps) * self.weight + self.bias
        else:
            return (values - mean)/(var.sqrt() + self.eps)

    def forward(self, inr: DiscretizedINR) -> DiscretizedINR:
        new_inr = inr.create_derived_inr()
        if self.batchnorm:
            self.batch_normalize(new_inr)
        else:
            self.inst_normalize(new_inr)
        return new_inr
