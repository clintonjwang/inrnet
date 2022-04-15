import torch
nn = torch.nn
F = nn.functional

class Linear(nn.Module):
    def __init__(self, in_C, out_C, r):
        super().__init__()
        self.r = r
        self.K = nn.Sequential(nn.Linear(2,16), nn.ReLU(inplace=True), nn.Linear(16,in_C*out_C))

    def forward(self, inr):
        (inr(x-dx,y-dy) * K(dx,dy)).reshape(-1, in_C, out_C).sum(1)
        return inr.integral()
