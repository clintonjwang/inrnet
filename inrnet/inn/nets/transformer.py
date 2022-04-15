import torch
nn = torch.nn
F = nn.functional

from inrnet import inn


class AttnNet(nn.Module):
    def __init__(self, out_ch, in_ch=3, spatial_dim=2, C=512):
        super().__init__()
        self.layer1 = SelfAttnINRLayer(in_ch, out_ch, spatial_dim=spatial_dim)
        self.layer2 = SelfAttnINRLayer(C, out_ch, spatial_dim=spatial_dim)

    def forward(self, inr):
        x = self.layer1(inr)
        x = F.silu(self.layer2(x), inplace=True) #torch.cat([x,t], dim=1)
        x = F.silu(self.layer3(x), inplace=True)
        return self.layer4(x)

