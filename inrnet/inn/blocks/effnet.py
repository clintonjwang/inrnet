import torch, pdb
nn = torch.nn
F = nn.functional
from functools import partial

from inrnet import inn

def translate_SE(discrete_se):
    sq,in_ = discrete_se.fc1.weight.shape[:2]
    cont_se = SqueezeExcitation(in_,sq) # need to change class to remove typing
    cont_se.fc1.weight.data = discrete_se.fc1.weight.data.squeeze(-1).squeeze(-1)
    cont_se.fc2.weight.data = discrete_se.fc2.weight.data.squeeze(-1).squeeze(-1)
    cont_se.fc1.bias.data = discrete_se.fc1.bias.data
    cont_se.fc2.bias.data = discrete_se.fc2.bias.data
    cont_se.activation = discrete_se.activation
    cont_se.scale_activation = discrete_se.scale_activation
    return cont_se

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, squeeze_channels,
            activation=nn.ReLU, scale_activation=nn.Sigmoid):
        super().__init__()
        self.avgpool = inn.GlobalAvgPool()
        self.fc1 = nn.Linear(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Linear(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, inr):
        scale = self.avgpool(inr)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def integrator(self, values):
        scale = values.mean(0, keepdim=True)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return values * self.scale_activation(scale)

    def forward(self, inr):
        # scale = self._scale(inr)
        # return inr * scale
        new_inr = inr.create_derived_inr()
        new_inr.integrator = self.integrator
        return new_inr
