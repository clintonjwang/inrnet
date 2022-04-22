
import torch, math
nn = torch.nn
F = nn.functional

class DiffusionINR2INR(nn.Module):
    def __init__(self, inr_size, t_dim=64, C=512):
        super().__init__()
        if t_dim > 0:
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(t_dim),
                nn.Linear(t_dim, t_dim * 4),
                nn.GELU(),
                nn.Linear(t_dim * 4, t_dim)
            )
            self.layer2 = nn.Linear(C+t_dim, C)
            self.layer4 = nn.Linear(C+t_dim, inr_size)
        else:
            self.layer2 = nn.Linear(C, C)
            self.layer4 = nn.Linear(C, inr_size)
        self.layer1 = nn.Linear(inr_size, C)
        self.bn = nn.BatchNorm1d(C)
        self.layer3 = nn.Linear(C, C)

    def forward(self, inr_params, time=None):
        if time is None:
            x = F.silu(self.layer1(inr_params), inplace=True)
            x = F.silu(self.layer2(x), inplace=True) + x
            x = self.bn(F.silu(self.layer3(x), inplace=True)) + x
            return self.layer4(x) + inr_params
        else:
            t = self.time_mlp(time)
            x = F.silu(self.layer1(inr_params), inplace=True)
            x = F.silu(self.layer2(torch.cat([x,t], dim=1)), inplace=True) + x
            x = self.bn(F.silu(self.layer3(x), inplace=True)) + x
            return self.layer4(torch.cat([x,t], dim=1)) + inr_params


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
