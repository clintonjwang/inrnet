import torch, pdb, math
import numpy as np
from functools import partial
nn = torch.nn
F = nn.functional

from inrnet import inn
from inrnet.inn import functional as inrF

def translate_convnext_model(input_shape, n_stages=3):
    extrema = ((-1,1),(-1,1))
    current_shape = input_shape
    sd = torch.load('/data/vision/polina/users/clintonw/code/diffcoord/temp/upernet_convnext.pth')['state_dict']

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
        nn.Sequential(cv, norm), current_shape, extrema)


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
        nn.Sequential(cv, norm), current_shape, extrema)

    stage0_blocks = []
    for d in range(3):
        root = f'backbone.stages.0.{d}.'
        bb_st_d_sd = {k[len(root):]:v for k,v in sd.items() if k.startswith(root)}
        stage0_blocks.append(translate_convnext_block(bb_st_d_sd, current_shape, extrema))
    stage0 = nn.Sequential(stem, *stage0_blocks)

    stages = [stage0]
    for stage in range(1,n_stages):
        root = 'backbone.downsample_layers.0.'
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


    norms = []
    for i in n_stages:
        ch = sd[f'backbone.norm{i}.weight'].size(0)
        norm = nn.InstanceNorm2d(ch, affine=True)
        norm.weight.data = sd[f'backbone.norm{i}.weight']
        norm.bias.data = sd[f'backbone.norm{i}.bias']
        norms.append(inn.conversion.translate_simple_layer(norm))

    root = 'decode_head.lateral_convs.'
    dec_lateral_sd = {k[len(root):]:v for k,v in sd.items() if k.startswith(root)}
    lateral_convs = []
    for i in range(n_stages-1):
        out_, in_ = dec_lateral_sd[f'{i}.conv.weight'].shape[:2]
        cv = inn.ChannelMixer(in_, out_, bias=False)
        cv.weight.data = dec_lateral_sd[f'{i}.conv.weight'].squeeze()
        norm = nn.BatchNorm2d(ch)
        norm.weight.data = dec_lateral_sd[f'{i}.bn.weight']
        norm.bias.data = dec_lateral_sd[f'{i}.bn.bias']
        norm.running_mean.data = dec_lateral_sd[f'{i}.bn.running_mean']
        norm.running_var.data = dec_lateral_sd[f'{i}.bn.running_var']
        norm = inn.conversion.translate_simple_layer(norm)
        act = inn.ReLU()
        lateral_convs.append(nn.Sequential(cv, norm, act))

    root = 'decode_head.fpn_convs.'
    dec_fpn_sd = {k[len(root):]:v for k,v in sd.items() if k.startswith(root)}
    fpn_convs = []
    for i in range(n_stages-1):
        out_, in_ = dec_lateral_sd[f'{i}.conv.weight'].shape[:2]
        cv = nn.Conv2d(in_, out_, 3, padding=1, bias=False)
        cv.weight.data = dec_lateral_sd[f'{i}.conv.weight']
        norm = nn.BatchNorm2d(ch)
        norm.weight.data = dec_lateral_sd[f'{i}.bn.weight']
        norm.bias.data = dec_lateral_sd[f'{i}.bn.bias']
        norm.running_mean.data = dec_lateral_sd[f'{i}.bn.running_mean']
        norm.running_var.data = dec_lateral_sd[f'{i}.bn.running_var']
        norm = inn.conversion.translate_simple_layer(norm)
        act = inn.ReLU()
        fpn_convs.append(nn.Sequential(cv, norm, act))

    root = 'decode_head.fpn_bottleneck.'
    fpn_bot_sd = {k[len(root):]:v for k,v in sd.items() if k.startswith(root)}
    fpn_bottleneck

    seg_cls = inn.Conv(num_classes)

    decoder = nn.ModuleDict(lateral_convs, fpn_convs)

    # root = 'auxiliary_head.'
    # aux_sd = {k[len(root):]:v for k,v in sd.items() if k.startswith(root)}
    # w = aux_sd['conv_seg.weight']
    # aux_sd['conv_seg.bias']

    convnext = ConvNeXt(stages, norms, decoder)


class ConvNeXt(nn.Module):
    def __init__(self, stages, norms, decoder):
        super().__init__()
        self.num_classes = decoder.num_classes
        self.stages = nn.ModuleList(stages)
        self.norms = nn.ModuleList(norms)
        self.decoder = decoder
    def integrator(self, values):
        x = self.stages[0](values)
        out0 = self.norms[0](x)
        x = self.stages[1](x)
        out1 = self.norms[1](x)
        x = self.stages[2](x)
        out2 = self.norms[2](x)
        return decoder(out0, out1, out2)
    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.set_integrator(self.integrator, 'ConvNeXt')
        new_inr.channels = self.num_classes
        return new_inr

class Decoder(nn.Module):
    def __init__(self, lateral_convs, fpn_convs, fpn_bottleneck, seg_cls):
        super().__init__()
        self.lateral_convs = lateral_convs
        self.fpn_convs = fpn_convs
        self.fpn_bottleneck = fpn_bottleneck
        self.seg_cls = seg_cls
        self.up2 = inn.upsample.Upsample(2, mode='bilinear')
        self.up4 = inn.upsample.Upsample(4, mode='bilinear')
    def integrator(self, out0, out1, out2):
        l1 = self.lateral_convs[0](out0)
        l1 = self.lateral_convs[1](out1)
        l1 = self.lateral_convs[2](out2)
        return seg



def translate_convnext_block(state_dict, current_shape, extrema):
    mid,in_ = state_dict['pointwise_conv1.weight'].shape
    depthwise_conv = nn.Conv2d(in_, in_, kernel_size=7, padding=3, groups=in_)
    depthwise_conv.weight.data = state_dict['depthwise_conv.weight']
    depthwise_conv.bias.data = state_dict['depthwise_conv.bias']
    depthwise_conv = inn.conversion.translate_strided_layer(norm, current_shape, extrema)[0]

    norm = nn.InstanceNorm2d(in_, affine=True)
    norm.weight.data = state_dict['norm.weight']
    norm.bias.data = state_dict['norm.bias']
    norm = inn.conversion.translate_simple_layer(norm)
    
    pointwise_conv1 = inn.ChannelMixer(in_, mid, bias=True)
    pointwise_conv1.weight.data = state_dict['pointwise_conv1.weight']
    pointwise_conv1.bias.data = state_dict['pointwise_conv1.bias']

    pointwise_conv2 = inn.ChannelMixer(mid_, in_, bias=True)
    pointwise_conv2.weight.data = state_dict['pointwise_conv2.weight']
    pointwise_conv2.bias.data = state_dict['pointwise_conv2.bias']

    layers = nn.Sequential(depthwise_conv, norm, pointwise_conv1, inn.GeLU(), pointwise_conv2)
    return CNBlock(layers, state_dict['gamma'])


class CNBlock(nn.Module):
    def __init__(self, layers, gamma):
        super().__init__()
        self.layers = layers
        self.gamma = nn.Parameter(gamma)
        #self.drop_path_rate = 0
    def forward(self, inr, inr=None):
        return inr + self.layers(inr.create_derived_inr())*self.gamma
