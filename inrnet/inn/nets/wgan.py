import torch, pdb, math
import numpy as np
from functools import partial
nn = torch.nn
F = nn.functional

from inrnet import inn
from inrnet.inn import functional as inrF

#G_layers=16, D_layers=14
def translate_wgan_model(G_layers=10, D_layers=8, mlp=128, mlp_ratio=1.):
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
    norm = inn.conversion.translate_simple_layer(norm)
    noise2inr = NoiseToINR(linear=fc, norm=norm, act=inn.ReLU(), bias=n2f_sd['bias'].squeeze()[:out_])

    current_shape = (4,4)
    extrema = ((-1,1),(-1,1))
    layers = []
    for i in range(G_layers):
        root = f'conv_blocks.{i}'
        try:
            out_, in_ = G_sd[f'{root}.conv.weight'].shape[:2]
            out_//=2
            in_//=2
            if in_ >= mlp:
                cv = inn.MLPConv(in_, out_, [k*mlp_ratio for k in inn.get_kernel_size(current_shape, extrema)])
            else:
                cv = nn.Conv2d(in_, out_, 3, padding=1)
                cv.weight.data = G_sd[f'{root}.conv.weight'][:out_, :in_]
                cv = inn.conversion.translate_strided_layer(cv, current_shape, extrema)[0]
            norm = nn.BatchNorm2d(out_)
            norm.weight.data = G_sd[f'{root}.bn.weight'][:out_]
            norm.bias.data = G_sd[f'{root}.bn.bias'][:out_] + G_sd[f'{root}.conv.bias'][:out_]
            norm.running_mean.data = G_sd[f'{root}.bn.running_mean'][:out_]
            norm.running_var.data = G_sd[f'{root}.bn.running_var'][:out_]
            norm = inn.conversion.translate_simple_layer(norm)
            act = inn.ReLU()
            layers.append(nn.Sequential(cv, norm, act))
        except KeyError:
            up, current_shape, extrema = inn.layers.upsample.translate_upsample(
                nn.Upsample(scale_factor=2, mode='nearest'), current_shape, extrema)
            layers.append(up)

    root = 'to_rgb.'
    rgb_sd = {k[len(root):]:v for k,v in G_sd.items() if k.startswith(root)}
    out_, in_ = rgb_sd['conv.weight'].shape[:2]
    in_ *= 2
    conv1x1 = inn.ChannelMixer(in_, out_, bias=True)
    conv1x1.weight.data[:,:in_//2] = rgb_sd['conv.weight'].squeeze()#[:,:in_//2]
    conv1x1.bias.data = rgb_sd['conv.bias']
    to_rgb = nn.Sequential(conv1x1, inn.Tanh())

    G = Generator(noise2inr=noise2inr, layers=nn.Sequential(*layers), to_rgb=to_rgb)


    root = 'from_rgb.'
    rgb_sd = {k[len(root):]:v for k,v in D_sd.items() if k.startswith(root)}
    out_, in_ = rgb_sd['conv.weight'].shape[:2]
    conv1x1 = inn.ChannelMixer(in_, out_, bias=True)
    conv1x1.weight.data = rgb_sd['conv.weight'].squeeze()
    conv1x1.bias.data = rgb_sd['conv.bias']
    from_rgb = nn.Sequential(conv1x1, inn.LeakyReLU(0.2))
    layers = []
    for i in range(D_layers):
        root = f'conv_blocks.{i}'
        try:
            out_, in_ = D_sd[f'{root}.conv.weight'].shape[:2]
            if in_ >= mlp:
                cv = inn.MLPConv(in_, out_, [k*mlp_ratio for k in inn.get_kernel_size(current_shape, extrema)])
            else:
                cv = nn.Conv2d(in_, out_, 3, padding=1)
                cv.weight.data = D_sd[f'{root}.conv.weight'][:out_, :in_]
                cv = inn.conversion.translate_strided_layer(cv, current_shape, extrema)[0]
            norm = nn.InstanceNorm2d(out_, affine=True)
            norm.weight.data = D_sd[f'{root}.GN.weight'][:out_]
            norm.bias.data = D_sd[f'{root}.GN.bias'][:out_] + D_sd[f'{root}.conv.bias'][:out_]
            norm = inn.conversion.translate_simple_layer(norm)
            act = inn.LeakyReLU(0.2)
            layers.append(nn.Sequential(cv, norm, act))
        except KeyError:
            down, current_shape, extrema = inn.conversion.translate_strided_layer(
                nn.AvgPool2d(kernel_size=2, stride=2), current_shape, extrema)
            layers.append(down)

    root = 'decision.'
    dec_sd = {k[len(root):]:v for k,v in D_sd.items() if k.startswith(root)}
    out_, in_ = dec_sd['conv.conv.weight'].shape[:2]
    in_ //= 2
    cv = inn.MLPConv(in_, out_, [k*mlp_ratio for k in inn.get_kernel_size(current_shape, extrema, k=4)])
    # out_, in_ = dec_sd['conv.conv.weight'].shape[:2]
    # cv = nn.Conv2d(in_, out_, 4, padding=1, bias=True)
    # cv.weight.data = D_sd['conv.conv.weight']
    # cv.bias.data = D_sd['conv.conv.bias']
    # cv, current_shape, extrema = inn.conversion.translate_strided_layer(cv, current_shape, extrema)
    norm = nn.InstanceNorm2d(out_, affine=True)
    norm.weight.data = dec_sd['conv.GN.weight'][:out_]
    norm.bias.data = dec_sd['conv.GN.bias'][:out_] + dec_sd[f'conv.conv.bias'][:out_]
    norm = inn.conversion.translate_simple_layer(norm)
    act = inn.LeakyReLU(.2)
    out_, in_ = dec_sd['linear.weight'].shape
    fc = nn.Linear(in_, out_, bias=True)
    fc.weight.data = dec_sd['linear.weight']
    fc.bias.data = dec_sd['linear.bias']
    pool = inn.AdaptiveAvgPoolSequence((2,2), fc, extrema=extrema)
    decision = nn.Sequential(cv, norm, act, pool)

    D = Discriminator(from_rgb=from_rgb, layers=nn.Sequential(*layers), decision=decision)
    return G,D



class Discriminator(nn.Module):
    def __init__(self, from_rgb, layers, decision):
        super().__init__()
        self.from_rgb = from_rgb
        self.layers = layers
        self.decision = decision
    def __len__(self):
        return 3
    def __getitem__(self, ix):
        if ix == 0:
            return self.from_rgb
        elif ix == 1:
            return self.layers
        elif ix == 2:
            return self.decision
        raise IndexError
    def __iter__(self):
        yield self.from_rgb
        yield self.layers
        yield self.decision
    def forward(self, inr):
        inr = self.from_rgb(inr)
        inr = self.layers(inr)
        return self.decision(inr)


class Generator(nn.Module):
    def __init__(self, noise2inr, layers, to_rgb):
        super().__init__()
        self.noise2inr = noise2inr
        self.layers = layers
        self.to_rgb = to_rgb
    def __len__(self):
        return 3
    def __getitem__(self, ix):
        if ix == 0:
            return self.noise2inr
        elif ix == 1:
            return self.layers
        elif ix == 2:
            return self.to_rgb
        raise IndexError
    def __iter__(self):
        yield self.noise2inr
        yield self.layers
        yield self.to_rgb
    def forward(self, noise):
        inr = self.noise2inr(noise)
        inr = self.layers(inr)
        return self.to_rgb(inr)

class NoiseToINR(nn.Module):
    def __init__(self, linear, norm, act, bias):
        super().__init__()
        self.linear = linear
        self.act = act
        self.norm = norm
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        assert x.ndim == 2
        x = torch.reshape(self.linear(x), (x.size(0), -1, 4, 4))
        inrs = inn.produce_inr(x)
        inrs += self.bias
        return self.norm(self.act(inrs))
