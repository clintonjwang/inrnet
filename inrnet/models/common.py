import os, pdb, torch, gc
osp = os.path
nn = torch.nn
F = nn.functional

def conv_bn_relu(in_, out_):
    return nn.Sequential(nn.Conv2d(in_, out_, 3,1,1, bias=False),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True))

def Conv2(in_channels, out_dims):
    layers = [conv_bn_relu(in_channels, 32),
        nn.MaxPool2d(2),
        conv_bn_relu(32, 64),
        nn.AdaptiveAvgPool2d(output_size=1), nn.Flatten(1),
        nn.Linear(64, out_dims)]
    for l in layers:
        if hasattr(l, 'weight'):
            nn.init.kaiming_uniform_(l.weight)
        if hasattr(l, 'bias'):
            nn.init.zeros_(l.bias)
    return nn.Sequential(*layers)

def Conv5(in_channels, out_dims):
    layers = [conv_bn_relu(in_channels, 32),
        nn.MaxPool2d(2),
        conv_bn_relu(32, 64),
        conv_bn_relu(64, 64),
        nn.MaxPool2d(2),
        conv_bn_relu(64, 64),
        conv_bn_relu(64, 64),
        nn.AdaptiveAvgPool2d(output_size=1), nn.Flatten(1),
        nn.Linear(64, out_dims)]
    for l in layers:
        if hasattr(l, 'weight'):
            nn.init.kaiming_uniform_(l.weight)
        if hasattr(l, 'bias'):
            nn.init.zeros_(l.bias)
    return nn.Sequential(*layers)

# def conv1d_bn_relu(in_, out_):
#     return nn.Sequential(nn.Conv2d(in_, out_, 3,1,1, bias=False),
#         nn.BatchNorm2d(out_),
#         nn.ReLU(inplace=True))

# def SimpleConv1d(in_channels, out_dims):
#     layers = [conv1d_bn_relu(in_channels, 32),
#         nn.MaxPool1d(2),
#         conv1d_bn_relu(32, 64),
#         nn.AdaptiveAvgPool1d(output_size=1), nn.Flatten(1),
#         nn.Linear(64, out_dims)]
#     for l in out_layers:
#         if hasattr(l, 'weight'):
#             nn.init.kaiming_uniform_(l.weight)
#         if hasattr(l, 'bias'):
#             nn.init.zeros_(l.bias)
#     return nn.Sequential(*layers)

class SimpleSeg(nn.Module):
    def __init__(in_channels, out_dims):
        super().__init__()
        self.in_channels = in_channels
        self.out_dims = out_dims
        self.layers = nn.Sequential(
            conv_bn_relu(in_channels, 32),
            nn.MaxPool2d(2),
            conv_bn_relu(32, 64),
            nn.Upsample2d(2, mode='bilinear'),
            *out_layers)
        self.last = nn.Conv2d(64, out_dims,1)
        for l in self.layers+[self.last]:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
            if hasattr(l, 'bias'):
                nn.init.zeros_(l.bias)
    def forward(self, x):
        self.last

# def SimpleG():