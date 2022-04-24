import torch, pdb
nn = torch.nn

from inrnet import inn
from inrnet.inn.blocks.effnet import translate_SE
from torchvision.ops.misc import SqueezeExcitation
from torchvision.models.efficientnet import MBConv

def translate_discrete_model(discrete_model, input_shape):
    extrema = ((-1,1),(-1,1))
    #if isinstance(layer, nn.modules.conv._ConvNd) or isinstance(layer, nn.modules.pooling._MaxPoolNd):
    InrNet, output_shape, extrema = translate_sequential_layer(discrete_model, input_shape, extrema)
    return InrNet.cuda(), output_shape


def translate_sequential_layer(layers, current_shape, extrema):
    cont_layers = []
    for layer in layers:
        if isinstance(layer, nn.modules.pooling._AdaptiveAvgPoolNd):
            cont_layer, current_shape, extrema = inn.GlobalAvgPool(), None, None
            break

        elif isinstance(layer, nn.modules.conv._ConvNd) or isinstance(layer, nn.modules.pooling._MaxPoolNd):
            cont_layer, current_shape, extrema = inn.conversion.translate_strided_layer(
                layer, current_shape, extrema)

        elif isinstance(layer, nn.Sequential):
            cont_layer, current_shape, extrema = inn.conversion.translate_sequential_layer(
                layer, current_shape, extrema)

        elif isinstance(layer, MBConv):
            if layer.use_res_connect:
                cont_layer = inn.blocks.ResBlock(translate_sequential_layer(
                    layer.block, current_shape, extrema)[0])
            else:
                cont_layer, current_shape, extrema = translate_sequential_layer(layer.block, current_shape, extrema)

        else:
            cont_layer = inn.conversion.translate_simple_layer(layer)
        cont_layers.append(cont_layer)
    remaining_layers = []
    for ix in range(len(cont_layers), len(layers)):
        remaining_layers.append(layers[ix])

    cont_sequence = nn.Sequential(*cont_layers, *remaining_layers)
    return cont_sequence, current_shape, extrema

def translate_simple_layer(layer):
    if layer.__class__ in (nn.ReLU, nn.LeakyReLU, nn.SiLU, nn.GELU):
        return inn.translate_activation(layer)

    elif isinstance(layer, nn.modules.batchnorm._BatchNorm):
        return inn.translate_bn(layer)

    elif isinstance(layer, SqueezeExcitation):
        return translate_SE(layer)

    else:
        raise NotImplementedError(layer.__class__)


def translate_strided_layer(layer, input_shape, extrema, **kwargs):
    if isinstance(layer, nn.Conv2d):
        if layer.weight.shape[-2:] == (1,1):
            return inn.translate_conv1x1(layer), input_shape, extrema
        else:
            return inn.translate_conv2d(layer, input_shape=input_shape, extrema=extrema, **kwargs)

    elif isinstance(layer, nn.MaxPool2d):
        return inn.translate_pool(layer, input_shape=input_shape, extrema=extrema, **kwargs)

    else:
        raise NotImplementedError(layer.__class__)
