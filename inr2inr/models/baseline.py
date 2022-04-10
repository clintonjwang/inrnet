import torch
nn = torch.nn
F = nn.functional


class INR2INR(nn.Module):
    def __init__(self, in_dim, out_dim, t_dim=0, C=256):
        super().__init__()
        if t_dim > 0:
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(t_dim),
                nn.Linear(t_dim, t_dim * 4),
                nn.GELU(),
                nn.Linear(t_dim * 4, t_dim)
            )
        else:
            self.time_mlp = None
            self.layer1 = nn.Linear(in_dim, C)
            self.layer2 = nn.Linear(C, C)
            # self.bn = nn.BatchNorm1d(C)
            self.layer3 = nn.Linear(C, C)
            self.layer4 = nn.Linear(C, out_dim)

    def forward(self, x, time):
        # t = self.time_mlp(time) if exists(self.time_mlp) else None
        x = F.relu(self.layer1(x), inplace=True)
        x = F.relu(self.layer2(x), inplace=True) #torch.cat([x,t], dim=1)
        x = F.relu(self.layer3(x), inplace=True)
        return self.layer4(x)

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