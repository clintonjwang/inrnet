import torch, pdb
nn = torch.nn
F = nn.functional

from inrnet import inn

def conv_norm_act(in_, out_, kernel_size=(.1,.1), **kwargs):
    act_layer = inn.get_activation_layer(kwargs.pop("activation", "swish"))
    cv = inn.MLPConv(in_, out_, kernel_size=kernel_size, **kwargs)
    # cv.mask_tracker = True
    return nn.Sequential(cv,
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
        down_ratio = kwargs.pop("down_ratio", 1)
        super().__init__(nn.Sequential(conv_norm_act(C, C, down_ratio=1, **kwargs),
            conv_norm_act(C, C, down_ratio=down_ratio, **kwargs)))
    # def get_convs(self):
    #     return self.sequential[0][0], self.sequential[1][0]
