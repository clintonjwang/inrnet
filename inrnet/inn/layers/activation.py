import torch
nn = torch.nn
F = nn.functional

def translate_activation(layer):
    if isinstance(layer, nn.ReLU):
        layer = ReLU()
    elif isinstance(layer, nn.LeakyReLU):
        layer = LeakyReLU(layer.negative_slope)
    elif isinstance(layer, nn.SiLU):
        layer = SiLU()
    elif isinstance(layer, nn.GELU):
        layer = GELU()
    else:
        raise NotImplementedError
    return layer

def get_activation_layer(type):
    if type is None:
        return nn.Identity()
    type = type.lower()
    if type == "relu":
        return ReLU()
    elif type == "leakyrelu":
        return LeakyReLU(.1)
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
    def __init__(self, negative_slope=.1):
        super().__init__()
        self.negative_slope = negative_slope
    def forward(self, inr):
        inr.add_modification(nn.LeakyReLU(self.negative_slope, inplace=True))
        return inr
class GELU(nn.Module):
    def forward(self, inr):
        inr.add_modification(nn.GELU())
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
