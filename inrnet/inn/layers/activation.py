import torch
nn = torch.nn
F = nn.functional

def get_activation_layer(type):
    type = type.lower()
    if type == "relu":
        return ReLU()
    elif type == "leakyrelu":
        return ReLU()
    elif type == "gelu":
        return GELU()
    elif type == "swish":
        return SiLU()
    elif type == "tanh":
        return Tanh()
    elif type == "sigmoid":
        return Sigmoid()
    else:
        raise NotImplementedError
        
class ReLU(nn.Module):
    def forward(self, inr):
        inr.add_modification(nn.ReLU(inplace=True))
        return inr
class LeakyReLU(nn.Module):
    def forward(self, inr):
        inr.add_modification(nn.LeakyReLU(inplace=True))
        return inr
class GELU(nn.Module):
    def forward(self, inr):
        inr.add_modification(nn.GELU(inplace=True))
        return inr
class SiLU(nn.Module):
    def forward(self, inr):
        inr.add_modification(nn.SiLU(inplace=True))
        return inr
class Tanh(nn.Module):
    def forward(self, inr):
        inr.add_modification(nn.Tanh())
        return inr
class Sigmoid(nn.Module):
    def forward(self, inr):
        inr.add_modification(nn.Sigmoid())
        return inr
