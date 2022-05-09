import torch, pdb, math
import numpy as np
from functools import partial
nn = torch.nn
F = nn.functional

from inrnet import inn
from inrnet.inn import functional as inrF
from inrnet.inn.layers.reshape import produce_inr


def translate_wgan_model(n_blocks=16, mlp=128, mlp_ratio=1.5):
    sd = torch.load('/data/vision/polina/users/clintonw/code/diffcoord/temp/wgan.pth')['state_dict']
    root = 'generator.'
    G_sd = {k[len(root):]:v for k,v in sd.items() if k.startswith(root)}
    root = 'discriminator.'
    D_sd = {k[len(root):]:v for k,v in sd.items() if k.startswith(root)}
    
    root = 'noise2feat.'
    n2f_sd = {k[len(root):]:v for k,v in G_sd.items() if k.startswith(root)}
    out_, noise_size = n2f_sd['linear.weight'].shape
    fc = nn.Linear(noise_size, out_, bias=False)
    fc.weight.data = n2f_sd['linear.weight']
    norm = nn.BatchNorm2d(out_)
    norm.weight.data = n2f_sd['norm.weight']
    norm.bias.data = n2f_sd['norm.bias']
    norm.running_mean.data = n2f_sd['norm.running_mean']
    norm.running_var.data = n2f_sd['norm.running_var']
    norm = inn.conversion.translate_simple_layer(norm)
    noise2inr = WGANNoiseToINR(linear=fc, norm=norm, act=inn.ReLU(), bias=n2f_sd['bias'].squeeze())

    current_shape = (4,4)
    extrema = ((-1,1),(-1,1))
    layers = []
    for i in range(n_blocks):
        root = f'conv_blocks.{i}'
        try:
            out_, in_ = G_sd[f'{root}.conv.weight'].shape[:2]
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
    conv1x1 = inn.ChannelMixer(in_, out_, bias=True)
    conv1x1.weight.data = rgb_sd['conv.weight']
    conv1x1.bias.data = rgb_sd['conv.bias']
    to_rgb = nn.Sequential(conv1x1, inn.Tanh())

    G = Generator(noise2inr=noise2inr, layers=nn.Sequential(*layers), to_rgb=to_rgb)


    root = 'from_rgb.'
    rgb_sd = {k[len(root):]:v for k,v in D_sd.items() if k.startswith(root)}
    out_, in_ = rgb_sd['conv.weight'].shape[:2]
    conv1x1 = inn.ChannelMixer(in_, out_, bias=True)
    conv1x1.weight.data = rgb_sd['conv.weight']
    conv1x1.bias.data = rgb_sd['conv.bias']
    from_rgb = nn.Sequential(conv1x1, inn.LeakyReLU(0.2))
    nn.AvgPool2d(kernel_size=2, stride=2)
    layers = []
    for i in range(n_blocks):
        root = f'conv_blocks.{i}'
        try:
            out_, in_ = G_sd[f'{root}.conv.weight'].shape[:2]
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

    D = Discriminator(noise2inr=noise2inr, layers=nn.Sequential(layers), to_rgb=to_rgb)
    return G,D



class Discriminator(nn.Module):
    def __init__(self, in_channel, in_scale, conv_module_cfg=None):
        super().__init__()

class Discriminator(nn.Module):
    _default_conv_module_cfg = dict(
        conv_cfg=None,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
        norm_cfg=dict(type='LN2d'),
        order=('conv', 'norm', 'act'))
    _default_upsample_cfg = dict(type='nearest', scale_factor=2)
    def __init__(self, in_channel, in_scale, conv_module_cfg=None):
        super().__init__()
        log2scale = int(np.log2(self.in_scale))
        for i in range(log2scale, 2, -1):
            self.conv_blocks.append(
                ConvLNModule(
                    self._default_channels_per_scale[str(2**i)],
                    self._default_channels_per_scale[str(2**i)],
                    feature_shape=(self._default_channels_per_scale[str(2**i)],
                                   2**i, 2**i),
                    **self.conv_module_cfg))
            self.conv_blocks.append(
                ConvLNModule(
                    self._default_channels_per_scale[str(2**i)],
                    self._default_channels_per_scale[str(2**(i - 1))],
                    feature_shape=(self._default_channels_per_scale[str(
                        2**(i - 1))], 2**i, 2**i),
                    **self.conv_module_cfg))
            self.conv_blocks.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.decision = WGANDecisionHead(
            self._default_channels_per_scale['4'],
            self._default_channels_per_scale['4'],
            1,
            act_cfg=dict(type='LeakyReLU', negative_slope=0.2),
            norm_cfg=self.conv_module_cfg['norm_cfg'])

    def forward(self, x):
        x = self.from_rgb(x)
        for conv in self.conv_blocks:
            x = conv(x)
        x = self.decision(x)
        return x


class WGANDecisionHead(nn.Module):
    """Module used in WGAN-GP to get the final prediction result with 4x4
    resolution input tensor in the bottom of the discriminator.
    Args:
        in_channels (int): Number of channels in input feature map.
        mid_channels (int): Number of channels in feature map after
            convolution.
        out_channels (int): The channel number of the final output layer.
        bias (bool, optional): Whether to use bias parameter. Defaults to True.
        act_cfg (dict, optional): Config for the activation layer. Defaults to
            dict(type='ReLU').
        out_act (dict, optional): Config for the activation layer of output
            layer. Defaults to None.
        norm_cfg (dict, optional): Config dict to build norm layer. Defaults to
            dict(type='LN2d').
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 bias=True,
                 act_cfg=dict(type='ReLU'),
                 out_act=None,
                 norm_cfg=dict(type='LN2d')):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.with_out_activation = out_act is not None

        # setup conv layer
        self.conv = ConvLNModule(
            in_channels,
            feature_shape=(mid_channels, 1, 1),
            kernel_size=4,
            out_channels=mid_channels,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            order=('conv', 'norm', 'act'))
        # setup linear layer
        self.linear = nn.Linear(
            self.mid_channels, self.out_channels, bias=bias)

        if self.with_out_activation:
            self.out_activation = build_activation_layer(out_act)

        self._init_weight()

    def forward(self, x):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x = self.conv(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.linear(x)
        if self.with_out_activation:
            x = self.out_activation(x)
        return x

    def _init_weight(self):
        """Initialize weights for the model."""
        nn.init.normal_(self.linear.weight, 0., 1.)
        nn.init.constant_(self.linear.bias, 0.)


class ConvLNModule():
    r"""ConvModule with Layer Normalization.
    In this module, we inherit default ``mmcv.cnn.ConvModule`` and deal with
    the situation that 'norm_cfg' is 'LN2d' or 'GN'. We adopt 'GN' as a
    replacement for layer normalization referring to:
    https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch/blob/master/module.py # noqa
    Args:
        feature_shape (tuple): The shape of feature map that will be.
    """

    def __init__(self, *args, feature_shape=None, **kwargs):
        if 'norm_cfg' in kwargs and kwargs['norm_cfg'] is not None and kwargs[
                'norm_cfg']['type'] in ['LN2d', 'GN']:
            nkwargs = deepcopy(kwargs)
            nkwargs['norm_cfg'] = None
            super().__init__(*args, **nkwargs)
            self.with_norm = True
            self.norm_name = kwargs['norm_cfg']['type']
            if self.norm_name == 'LN2d':
                norm = nn.LayerNorm(feature_shape)
                self.add_module(self.norm_name, norm)
            else:
                norm = nn.GroupNorm(1, feature_shape[0])
                self.add_module(self.norm_name, norm)
        else:
            super().__init__(*args, **kwargs)


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

class WGANNoiseToINR(nn.Module):
    def __init__(self, linear, norm, act, bias):
        super().__init__()
        self.linear = linear
        self.act = act
        self.norm = norm
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        assert x.ndim == 2
        x = torch.reshape(self.linear(x), (x.size(0), -1, 4, 4))
        inrs = produce_inr(x)
        inrs += self.bias
        inrs = self.activation(inr)
        inrs = self.norm(inr)
        return inrs
