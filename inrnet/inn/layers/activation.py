import torch
nn = torch.nn
F = nn.functional

class ReLU(nn.Module):
    def forward(self, inr):
        inr.add_modification(nn.ReLU(inplace=True))
        return inr
class LeakyReLU(nn.Module):
    def forward(self, inr):
        inr.add_modification(nn.LeakyReLU(inplace=True))
        return inr
