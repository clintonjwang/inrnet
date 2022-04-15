import torch
nn = torch.nn
F = nn.functional

class InstanceNorm(nn.Module):
    def __init__(self, channels=None, affine=False):
        # normalizes every channel
        super().__init__()
        if track_running_stats:
            raise NotImplementedError("TODO: track_running_stats")
        self.affine = affine
        if affine:
            self.learned_mean = nn.Parameter(torch.zeros(channels).float())
            self.learned_std = nn.Parameter(torch.ones(channels).float())

    def forward(self, inr):
        return inr.normalize(self)


class BatchNorm(nn.Module):
    def __init__(self, channels, momentum=0.1, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.learned_std = nn.Parameter(torch.empty(channels, **factory_kwargs))
        self.learned_mean = nn.Parameter(torch.empty(channels, **factory_kwargs))
        self.register_buffer('running_mean', torch.zeros(channels, **factory_kwargs))
        self.register_buffer('running_var', torch.ones(channels, **factory_kwargs))
        self.reset_parameters()
        self.momentum = momentum

    def reset_running_stats(self):
        self.running_mean.zero_()  # type: ignore[union-attr]
        self.running_var.fill_(1)  # type: ignore[union-attr]

    def reset_parameters(self):
        self.reset_running_stats()
        nn.init.ones_(self.learned_std)
        nn.init.zeros_(self.learned_mean)

    def forward(self, inr):
        return inr.normalize(self)

