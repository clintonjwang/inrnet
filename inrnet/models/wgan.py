import torch, pdb, math
import numpy as np
nn = torch.nn
F = nn.functional

from inrnet.models.common import Conv2, conv_bn_relu
from inrnet import inn

def Gan4(reshape):
    G = G4(in_dims=64, out_channels=1, reshape=reshape)
    D = Conv2(in_channels=1, out_dims=1, C=16)
    return G,D

class G4(nn.Module):
    def __init__(self, in_dims, out_channels, C=8, reshape=(7,7)):
        super().__init__()
        self.first = nn.Sequential(
            nn.Linear(in_dims, C*64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(C*64, C*32, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(C*32, C*reshape[0]*reshape[1], bias=True),
        )
        cv = nn.Conv2d(C, out_channels, 1, bias=True)
        self.reshape = reshape
        for l in [*self.first, cv]:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
            if hasattr(l, 'bias'):
                nn.init.zeros_(l.bias)

        layers = [
            # nn.Upsample(scale_factor=(2,2), mode='nearest'),
            # conv_bn_relu(C, C*4),
            # nn.Upsample(scale_factor=(2,2), mode='nearest'), #32x32
            # conv_bn_relu(C, C*2),
            cv, nn.Tanh()
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.reshape(self.first(x), (x.size(0), -1, *self.reshape))
        return self.layers(x)

#G_layers=16, D_layers=14
def simple_wgan(G_layers=10, D_layers=8):
    sd = torch.load('/data/vision/polina/users/clintonw/code/diffcoord/temp/wgan.pth')['state_dict']
    root = 'generator.'
    G_sd = {k[len(root):]:v for k,v in sd.items() if k.startswith(root)}
    root = 'discriminator.'
    D_sd = {k[len(root):]:v for k,v in sd.items() if k.startswith(root)}
    
    root = 'noise2feat.'
    n2f_sd = {k[len(root):]:v for k,v in G_sd.items() if k.startswith(root)}
    out_, noise_size = n2f_sd['linear.weight'].shape
    out_//=2
    fc = nn.Linear(noise_size, out_, bias=False)
    fc.weight.data = n2f_sd['linear.weight'][:out_]
    out_ = n2f_sd['norm.weight'].size(0)//2
    norm = nn.BatchNorm2d(out_)
    norm.weight.data = n2f_sd['norm.weight'][:out_]
    norm.bias.data = n2f_sd['norm.bias'][:out_]
    norm.running_mean.data = n2f_sd['norm.running_mean'][:out_]
    norm.running_var.data = n2f_sd['norm.running_var'][:out_]
    noise2feats = NoiseToFeats(linear=fc, norm=norm, act=nn.ReLU(), bias=n2f_sd['bias'][:,:out_])

    layers = []
    for i in range(G_layers):
        root = f'conv_blocks.{i}'
        try:
            out_, in_ = G_sd[f'{root}.conv.weight'].shape[:2]
            out_ //= 2
            in_ //= 2
            cv = nn.Conv2d(in_, out_, 3, padding=1)
            cv.weight.data = G_sd[f'{root}.conv.weight'][:out_, :in_]
            norm = nn.BatchNorm2d(out_)
            norm.weight.data = G_sd[f'{root}.bn.weight'][:out_]
            norm.bias.data = G_sd[f'{root}.bn.bias'][:out_] + G_sd[f'{root}.conv.bias'][:out_]
            norm.running_mean.data = G_sd[f'{root}.bn.running_mean'][:out_]
            norm.running_var.data = G_sd[f'{root}.bn.running_var'][:out_]
            act = nn.ReLU()
            layers.append(nn.Sequential(cv, norm, act))
        except KeyError:
            up = nn.Upsample(scale_factor=2, mode='nearest')
            layers.append(up)

    root = 'to_rgb.'
    rgb_sd = {k[len(root):]:v for k,v in G_sd.items() if k.startswith(root)}
    out_, in_ = rgb_sd['conv.weight'].shape[:2]
    in_ *= 2
    conv1x1 = nn.Conv2d(in_, out_, 1, bias=True)
    conv1x1.weight.data[:,:in_//2] = rgb_sd['conv.weight']
    conv1x1.bias.data = rgb_sd['conv.bias']
    to_rgb = nn.Sequential(conv1x1, nn.Tanh())

    G = Generator(noise2feats=noise2feats, layers=nn.Sequential(*layers), to_rgb=to_rgb)


    root = 'from_rgb.'
    rgb_sd = {k[len(root):]:v for k,v in D_sd.items() if k.startswith(root)}
    out_, in_ = rgb_sd['conv.weight'].shape[:2]
    conv1x1 = nn.Conv2d(in_, out_, 1, bias=True)
    conv1x1.weight.data = rgb_sd['conv.weight']
    conv1x1.bias.data = rgb_sd['conv.bias']
    from_rgb = nn.Sequential(conv1x1, nn.LeakyReLU(0.2))
    layers = []
    for i in range(D_layers):
        root = f'conv_blocks.{i}'
        try:
            out_, in_ = D_sd[f'{root}.conv.weight'].shape[:2]
            cv = nn.Conv2d(in_, out_, 3, padding=1)
            cv.weight.data = D_sd[f'{root}.conv.weight'][:out_, :in_]
            norm = nn.InstanceNorm2d(out_, affine=True)
            norm.weight.data = D_sd[f'{root}.GN.weight'][:out_]
            norm.bias.data = D_sd[f'{root}.GN.bias'][:out_] + D_sd[f'{root}.conv.bias'][:out_]
            act = nn.LeakyReLU(0.2)
            layers.append(nn.Sequential(cv, norm, act))
        except KeyError:
            down = nn.MaxPool2d(kernel_size=2, stride=2)
            layers.append(down)

    root = 'decision.'
    dec_sd = {k[len(root):]:v for k,v in D_sd.items() if k.startswith(root)}
    out_, in_ = dec_sd['conv.conv.weight'].shape[:2]
    in_ //= 2
    cv = nn.Conv2d(in_, out_, 4, bias=True)
    cv.weight.data = dec_sd['conv.conv.weight'][:,:in_]
    cv.bias.data = dec_sd['conv.conv.bias']
    norm = nn.InstanceNorm2d(out_, affine=True)
    norm.weight.data = dec_sd['conv.GN.weight'][:out_]
    norm.bias.data = dec_sd['conv.GN.bias'][:out_] + dec_sd[f'conv.conv.bias'][:out_]
    act = nn.LeakyReLU(.2)
    out_, in_ = dec_sd['linear.weight'].shape
    fc = nn.Linear(in_, out_, bias=True)
    fc.weight.data = dec_sd['linear.weight']
    fc.bias.data = dec_sd['linear.bias']
    decision = nn.Sequential(cv, norm, act, nn.AdaptiveAvgPool2d(1), nn.Flatten(1), fc)

    D = Discriminator(from_rgb=from_rgb, layers=nn.Sequential(*layers), decision=decision)
    return G,D



class Discriminator(nn.Module):
    def __init__(self, from_rgb, layers, decision):
        super().__init__()
        self.from_rgb = from_rgb
        self.layers = layers
        self.decision = decision
    def forward(self, x):
        x = self.from_rgb(x)
        x = self.layers(x)
        return self.decision(x)


class Generator(nn.Module):
    def __init__(self, noise2feats, layers, to_rgb):
        super().__init__()
        self.noise2feats = noise2feats
        self.layers = layers
        self.to_rgb = to_rgb
    def forward(self, noise):
        x = self.noise2feats(noise)
        x = self.layers(x)
        return self.to_rgb(x)

class NoiseToFeats(nn.Module):
    def __init__(self, linear, norm, act, bias):
        super().__init__()
        self.linear = linear
        self.act = act
        self.norm = norm
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        assert x.ndim == 2
        x = torch.reshape(self.linear(x), (x.size(0), -1, 4, 4)) + self.bias
        return self.norm(self.act(x))
