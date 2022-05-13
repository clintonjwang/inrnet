import torch, pdb, math
import numpy as np
from functools import partial
nn = torch.nn
F = nn.functional

from inrnet import inn
from inrnet.inn import functional as inrF

def translate_convnext_model(input_shape, n_stages=2, mlp=128):
    extrema = ((-1,1),(-1,1))
    current_shape = input_shape
    sd = torch.load('/data/vision/polina/users/clintonw/code/diffcoord/temp/upernet_convnext.pth')['state_dict']

    # all the top level layers need to be built before we can change shape/extrema
    root = 'decode_head.lateral_convs.'
    dec_lateral_sd = {k[len(root):]:v for k,v in sd.items() if k.startswith(root)}
    lateral_convs = []
    norms = []

    root = 'decode_head.fpn_convs.'
    dec_fpn_sd = {k[len(root):]:v for k,v in sd.items() if k.startswith(root)}
    fpn_convs = []

    root = 'decode_head.fpn_bottleneck.'
    fpn_bot_sd = {k[len(root):]:v for k,v in sd.items() if k.startswith(root)}
    out_, in_ = fpn_bot_sd['conv.weight'].shape[:2]
    in_ = 256
    out_ = 128
    if in_ >= mlp:
        cv = inn.MLPConv(in_, out_, inn.get_kernel_size(current_shape, extrema, k=5), groups=32)
    else:
        cv = nn.Conv2d(in_, out_, 3, padding=1, bias=False)
        cv.weight.data = fpn_bot_sd['conv.weight'][:out_, :in_]
        cv = inn.conversion.translate_strided_layer(cv, current_shape, extrema)[0]
    norm = nn.BatchNorm2d(out_)
    norm.weight.data = fpn_bot_sd['bn.weight'][:out_]
    norm.bias.data = fpn_bot_sd['bn.bias'][:out_]
    norm.running_mean.data = fpn_bot_sd['bn.running_mean'][:out_]
    norm.running_var.data = fpn_bot_sd['bn.running_var'][:out_]
    norm = inn.conversion.translate_simple_layer(norm)
    act = inn.ReLU()
    fpn_bottleneck = nn.Sequential(cv, norm, act)


    root = 'backbone.downsample_layers.0.'
    bb_ds_sd = {k[len(root):]:v for k,v in sd.items() if k.startswith(root)}
    out_,in_,kw,kh = bb_ds_sd['0.weight'].shape
    cv = nn.Conv2d(in_, out_, kernel_size=(4,4), stride=(2,2), padding=(1,1))
    cv.weight.data = bb_ds_sd['0.weight']
    cv.bias.data = bb_ds_sd['0.bias']
    norm = nn.InstanceNorm2d(out_, affine=True)
    norm.weight.data = bb_ds_sd['1.weight']
    norm.bias.data = bb_ds_sd['1.bias']
    stem, current_shape, extrema = inn.conversion.translate_sequential_layer(
        nn.Sequential(cv, norm), input_shape, extrema)

    stage0_blocks = []
    for d in range(3):
        root = f'backbone.stages.0.{d}.'
        bb_st_d_sd = {k[len(root):]:v for k,v in sd.items() if k.startswith(root)}
        stage0_blocks.append(translate_convnext_block(bb_st_d_sd, current_shape, extrema))
    stage0 = nn.Sequential(stem, *stage0_blocks)

    up0 = inn.upsample.translate_upsample(nn.Upsample(scale_factor=2, mode='bilinear'), current_shape, extrema)[0]

    stages = [stage0]
    for stage in range(1,n_stages+1):
        i = stage-1
        out_, in_ = dec_lateral_sd[f'{i}.conv.weight'].shape[:2]
        out_ //= 2
        cv = inn.ChannelMixer(in_, out_, bias=False)
        cv.weight.data = dec_lateral_sd[f'{i}.conv.weight'][:out_].squeeze()
        norm = nn.BatchNorm2d(out_)
        norm.weight.data = dec_lateral_sd[f'{i}.bn.weight'][:out_]
        norm.bias.data = dec_lateral_sd[f'{i}.bn.bias'][:out_]
        norm.running_mean.data = dec_lateral_sd[f'{i}.bn.running_mean'][:out_]
        norm.running_var.data = dec_lateral_sd[f'{i}.bn.running_var'][:out_]
        norm = inn.conversion.translate_simple_layer(norm)
        act = inn.ReLU()
        lateral_convs.append(nn.Sequential(cv, norm, act))

        out_, in_ = dec_fpn_sd[f'{i}.conv.weight'].shape[:2]
        in_ //= 2
        out_ //= 4
        if in_ >= mlp:
            cv = inn.MLPConv(in_, out_, inn.get_kernel_size(current_shape, extrema, k=5), groups=32)
        else:
            cv = nn.Conv2d(in_, out_, 3, padding=1, bias=False)
            cv.weight.data = dec_fpn_sd[f'{i}.conv.weight'][:out_, :in_]
            cv = inn.conversion.translate_strided_layer(cv, current_shape, extrema)[0]
        norm = nn.BatchNorm2d(out_)
        norm.weight.data = dec_fpn_sd[f'{i}.bn.weight'][:out_]
        norm.bias.data = dec_fpn_sd[f'{i}.bn.bias'][:out_]
        norm.running_mean.data = dec_fpn_sd[f'{i}.bn.running_mean'][:out_]
        norm.running_var.data = dec_fpn_sd[f'{i}.bn.running_var'][:out_]
        norm = inn.conversion.translate_simple_layer(norm)
        act = inn.ReLU()
        fpn_convs.append(nn.Sequential(cv, norm, act))

        if stage == n_stages:
            break

        # downsample_layers
        root = f'backbone.downsample_layers.{stage}.'
        bb_ds_sd = {k[len(root):]:v for k,v in sd.items() if k.startswith(root)}
        out_,in_,kw,kh = bb_ds_sd['1.weight'].shape
        norm = nn.InstanceNorm2d(in_, affine=True)
        norm.weight.data = bb_ds_sd['0.weight']
        norm.bias.data = bb_ds_sd['0.bias']
        cv = nn.Conv2d(in_, out_, kernel_size=(2,2), stride=(2,2))
        cv.weight.data = bb_ds_sd['1.weight']
        cv.bias.data = bb_ds_sd['1.bias']
        norm_ds, current_shape, extrema = inn.conversion.translate_sequential_layer(
            nn.Sequential(norm, cv), current_shape, extrema)

        stage_blocks = []
        for d in range(3):
            root = f'backbone.stages.{stage}.{d}.'
            bb_st_d_sd = {k[len(root):]:v for k,v in sd.items() if k.startswith(root)}
            stage_blocks.append(translate_convnext_block(bb_st_d_sd, current_shape, extrema))
        stages.append(nn.Sequential(norm_ds, *stage_blocks))
    
    up1 = inn.upsample.translate_upsample(nn.Upsample(scale_factor=2, mode='bilinear'), current_shape, extrema)[0]


    for i in range(n_stages):
        ch = sd[f'backbone.norm{i}.weight'].size(0)
        norm = nn.InstanceNorm2d(ch, affine=True)
        norm.weight.data = sd[f'backbone.norm{i}.weight']
        norm.bias.data = sd[f'backbone.norm{i}.bias']
        norms.append(inn.conversion.translate_simple_layer(norm))


    num_classes = 7
    seg_cls = inn.ChannelMixer(128, num_classes, bias=True)
    nn.init.kaiming_uniform_(seg_cls.weight, mode='fan_in')
    seg_cls.bias.data.zero_()
    decoder = Decoder(lateral_convs, fpn_convs, fpn_bottleneck, seg_cls, num_classes=num_classes, ups=(up0, up1))

    # root = 'auxiliary_head.'
    # aux_sd = {k[len(root):]:v for k,v in sd.items() if k.startswith(root)}
    # w = aux_sd['conv_seg.weight']
    # aux_sd['conv_seg.bias']

    convnext = ConvNeXt(stages, norms, decoder)
    return convnext



class ConvNeXt(nn.Module):
    def __init__(self, stages, norms, decoder):
        super().__init__()
        self.num_classes = decoder.num_classes
        self.stages = nn.ModuleList(stages)
        self.norms = nn.ModuleList(norms)
        self.decoder = decoder
    def forward(self, inr):
        # self = InrNet
        x = self.stages[0](inr)
        out0 = self.norms[0](x)
        x = self.stages[1](x)
        out1 = self.norms[1](x)
        new_inr = self.decoder(out0, out1)
        return new_inr
    def __len__(self):
        return 3
    def __getitem__(self, ix):
        if ix == 0:
            return self.stages
        elif ix == 1:
            return self.norms
        elif ix == 2:
            return self.decoder
    def __iter__(self):
        yield self.stages
        yield self.norms
        yield self.decoder

class Decoder(nn.Module):
    def __init__(self, lateral_convs, fpn_convs, fpn_bottleneck, seg_cls, num_classes, ups):
        super().__init__()
        self.lateral_convs = nn.ModuleList(lateral_convs)
        self.fpn_convs = nn.ModuleList(fpn_convs)
        self.num_classes = num_classes
        self.fpn_bottleneck = fpn_bottleneck
        self.num_classes = num_classes
        self.seg_cls = seg_cls
        self.ups = nn.ModuleList(ups)
        self.main_modules = [self.lateral_convs, self.fpn_convs, self.fpn_bottleneck, self.seg_cls]
    def forward(self, out0, out1):
        l1 = self.lateral_convs[1](out1)
        l0 = self.lateral_convs[0](out0) + self.ups[1](l1)
        f0 = self.fpn_convs[0](l0)
        f1 = self.fpn_convs[1](l1)
        fpns = f0.cat(self.ups[1](f1))
        top = self.fpn_bottleneck(self.ups[0](fpns))
        return self.seg_cls(top)
    def __len__(self):
        return len(self.main_modules)
    def __getitem__(self, ix):
        return self.main_modules[ix]
    def __iter__(self):
        for m in self.main_modules:
            yield m


def translate_convnext_block(state_dict, current_shape, extrema):
    mid,in_ = state_dict['pointwise_conv1.weight'].shape
    depthwise_conv = nn.Conv2d(in_, in_, kernel_size=7, padding=3, groups=in_)
    depthwise_conv.weight.data = state_dict['depthwise_conv.weight']
    depthwise_conv.bias.data = state_dict['depthwise_conv.bias']
    depthwise_conv = inn.conversion.translate_strided_layer(depthwise_conv, current_shape, extrema)[0]

    norm = nn.InstanceNorm2d(in_, affine=True)
    norm.weight.data = state_dict['norm.weight']
    norm.bias.data = state_dict['norm.bias']
    norm = inn.conversion.translate_simple_layer(norm)
    
    pointwise_conv1 = inn.ChannelMixer(in_, mid, bias=True)
    pointwise_conv1.weight.data = state_dict['pointwise_conv1.weight']
    pointwise_conv1.bias.data = state_dict['pointwise_conv1.bias']

    pointwise_conv2 = inn.ChannelMixer(mid, in_, bias=True)
    pointwise_conv2.weight.data = state_dict['pointwise_conv2.weight']
    pointwise_conv2.bias.data = state_dict['pointwise_conv2.bias']

    layers = nn.Sequential(depthwise_conv, norm, pointwise_conv1, inn.GELU(), pointwise_conv2)
    return CNBlock(layers, state_dict['gamma'])

def enable_cn_blocks(network):
    for m in network:
        if hasattr(m, '__iter__'):
            enable_cn_blocks(m)
        elif isinstance(m, CNBlock):
            m.drop = False


class CNBlock(nn.Module):
    def __init__(self, layers, gamma):
        super().__init__()
        self.layers = layers
        self.gamma = nn.Parameter(gamma*.5)
        self.drop = True
    def forward(self, inr):
        if self.drop:
            return inr
        else:
            return inr + self.layers(inr.create_derived_inr())*self.gamma
