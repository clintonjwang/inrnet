import torch
nn = torch.nn

from inrnet import inn

def Gan4(reshape):
    G = G4(in_dims=64, out_channels=1, reshape=reshape)
    D = D4(in_channels=1, C=32)
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
        cm = inn.ChannelMixer(C, out_channels, bias=True)
        for l in [*self.first, cm]:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
            if hasattr(l, 'bias'):
                nn.init.zeros_(l.bias)
        self.reshape = reshape

        #kwargs = dict(mid_ch=(16,16), N_bins=64) #i4
        kwargs = dict()#mid_ch=(16,32), N_bins=64) #i4b
        #activation='relu') #mid_ch=(16,16)
        layers = [
            # inn.blocks.conv_norm_act(C, C*2, kernel_size=(.5,.5)), #
            # inn.Upsample(4, spacing=(1/6,1/6), align_corners=True),
            # inn.blocks.conv_norm_act(C, C*2, kernel_size=(.85,.85), **kwargs), #5x5 kernel, 14x14
            # inn.PositionalEncoding(N=C),
            # inn.blocks.conv_norm_act(C, C, kernel_size=(.5,.5), **kwargs), #3x3 kernel, 14x14
            # inn.MLPConv(C, C, kernel_size=(.18,.18), **kwargs),
            # inn.BallConv(C,C, radius=.18)
            # inn.Upsample(4, spacing=(1/15,1/15), align_corners=True),
            # inn.blocks.ResConv(C, kernel_size=(.18,.18)),
            inn.blocks.conv_norm_act(C, C*16, kernel_size=(.31,.31), **kwargs),
            inn.blocks.conv_norm_act(C*16, C, kernel_size=(.18,.18), **kwargs),
            # inn.blocks.conv_norm_act(C*2, C, kernel_size=(.18,.18), **kwargs),
            # inn.ChannelNorm(C*2, batchnorm=False),
            # inn.ReLU(),
            # inn.blocks.conv_norm_act(C, C*2, kernel_size=(.2,.2), **kwargs), #32x32
            cm, inn.Tanh(),
        ]
        self.layers = nn.Sequential(*layers)
        x = torch.as_tensor((-1/8,-1/16,0,1/16,1/8))
        cvs = (self.layers[0][0], self.layers[1][0])
        #(self.layers[0][0][0],self.layers[0][1][0])#
        for cv in cvs:
            cv.register_buffer("grid_points", torch.dstack(torch.meshgrid(x,x, indexing='ij')).reshape(-1,2))
            cv.N_bins = 25
        #G.layers[1][0].mask_tracker

    def forward(self, x):
        x = torch.reshape(self.first(x), (x.size(0), -1, *self.reshape))
        inrs = inn.produce_inr(x)
        return self.layers(inrs)


class D4(nn.Module):
    def __init__(self, in_channels, out_dims=1, C=32):#, **kwargs):
        super().__init__()
        out_layers = nn.Sequential(nn.Linear(C*16*2, C*4), nn.ReLU(inplace=True),
            nn.Linear(C*4, 128), nn.ReLU(inplace=True),
            nn.Linear(128, out_dims))
        for l in out_layers:
            if hasattr(l, 'weight'):
                nn.init.kaiming_uniform_(l.weight)
                nn.init.zeros_(l.bias)
        kwargs = dict(mid_ch=(8,8), N_bins=64, mlp_type='normal')
        self.layers = [
            inn.blocks.conv_norm_act(in_channels, C, kernel_size=(.18,.18), down_ratio=.25, **kwargs),
            # inn.MaxPool((.19,.19), down_ratio=.25),
            # inn.blocks.conv_norm_act(C, C, kernel_size=(.38,.38), **kwargs),
            inn.blocks.conv_norm_act(C, C*2, kernel_size=(.38,.38), down_ratio=.25, **kwargs),
            inn.AdaptiveAvgPoolSequence((4,4), out_layers),
        ]
        self.layers = nn.Sequential(*self.layers)

    def get_convs(self):
        return self.layers[0][0], self.layers[2][0]
        #D.layers[2][0].mask_tracker

    def forward(self, inr):
        return self.layers(inr)


#G_layers=16, D_layers=14
def translate_wgan_model(G_layers=10, D_layers=8, mlp=128):
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
            out_ //= 2
            in_ //= 2
            if in_ >= mlp:
                cv = inn.MLPConv(in_, out_, inn.get_kernel_size(current_shape, extrema, k=5))
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
                cv = inn.MLPConv(in_, out_, inn.get_kernel_size(current_shape, extrema, k=5))
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
    cv = inn.MLPConv(in_, out_, inn.get_kernel_size(current_shape, extrema, k=5))
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
