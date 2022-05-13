import torch, pdb
nn = torch.nn
F = nn.functional

from inrnet import inn

def conv_norm_act(in_, out_, kernel_size=(.1,.1), **kwargs):
    act_layer = inn.get_activation_layer(kwargs.pop("activation", "swish"))
    return nn.Sequential(inn.MLPConv(in_, out_, kernel_size=kernel_size, **kwargs),
        inn.ChannelNorm(out_),
        act_layer,
    )

class ResBlock(nn.Module):
    def __init__(self, sequential):
        super().__init__()
        self.sequential = sequential
    def forward(self, inr):
        return inr + self.sequential(inr.create_derived_inr())
    def __getitem__(self, ix):
        return self.sequential[ix]
    def __len__(self):
        return len(self.sequential)
    def __iter__(self):
        return self.sequential.__iter__()

# class ResTest(nn.Module):
#     def __init__(self, C):
#         super().__init__()
#         self.block = nn.Sequential(nn.Conv2d(C,C,3,1,1,bias=False))
#     def forward(self, x):
#         return x + self.block(x)

class ResConv(ResBlock):
    def __init__(self, C, **kwargs):
        stride = kwargs.pop("stride", 0)
        super().__init__(nn.Sequential(conv_norm_act(C, C, stride=0, **kwargs),
            conv_norm_act(C, C, stride=stride, **kwargs)))
