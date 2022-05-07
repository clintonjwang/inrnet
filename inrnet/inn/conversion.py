import torch, pdb
nn = torch.nn

from inrnet import inn
from inrnet.inn.blocks.effnet import translate_SE, translate_fire
try:
    from torchvision.ops.misc import SqueezeExcitation
    from torchvision.models.efficientnet import EfficientNet, MBConv
    from torchvision.models.squeezenet import SqueezeNet, Fire
except ModuleNotFoundError:
    SqueezeExcitation = EfficientNet = MBConv = Fire = None

def translate_discrete_model(discrete_model, input_shape):
    extrema = ((-1,1),(-1,1))
    if isinstance(discrete_model, EfficientNet):
        discrete_model = nn.Sequential(discrete_model.features,
            discrete_model.avgpool, discrete_model.classifier)
    elif isinstance(discrete_model, SqueezeNet):
        discrete_model = nn.Sequential(discrete_model.features, discrete_model.classifier)
    InrNet, output_shape, extrema = translate_sequential_layer(discrete_model, input_shape, extrema)
    return InrNet.cuda(), output_shape


def translate_sequential_layer(layers, current_shape, extrema):
    cont_layers = []
    z = 0
    for layer_num,layer in enumerate(layers):
        if isinstance(layer, nn.modules.pooling._AdaptiveAvgPoolNd):
            # cont_layer, current_shape, extrema = inn.GlobalAvgPool(), None, None
            break

        elif isinstance(layer, nn.modules.padding._ReflectionPadNd) or isinstance(layer, nn.modules.dropout._DropoutNd):
            z += 1
            continue

        elif isinstance(layer, nn.Upsample):
            cont_layer, current_shape, extrema = inn.upsample.translate_upsample(
                layer, current_shape, extrema)

        elif isinstance(layer, nn.modules.conv._ConvNd) or isinstance(layer, nn.modules.pooling._MaxPoolNd):
            cont_layer, current_shape, extrema = translate_strided_layer(
                layer, current_shape, extrema)

        elif isinstance(layer, nn.Sequential):
            cont_layer, current_shape, extrema = translate_sequential_layer(
                layer, current_shape, extrema)

        elif isinstance(layer, Fire):
            cont_layer = translate_fire(layer, current_shape, extrema)

        elif isinstance(layer, MBConv):
            if layer.use_res_connect:
                cont_layer = inn.blocks.ResBlock(translate_sequential_layer(
                    layer.block, current_shape, extrema)[0])
            else:
                cont_layer, current_shape, extrema = translate_sequential_layer(layer.block, current_shape, extrema)

        else:
            cont_layer = translate_simple_layer(layer)

        cont_layers.append(cont_layer)

    remaining_layers = []
    for ix in range(layer_num+1+z, len(layers)):
        remaining_layers.append(layers[ix])
    if len(remaining_layers) > 0:
        cont_layers.append(inn.GlobalAvgPoolSequence(nn.Sequential(*remaining_layers)))
        current_shape, extrema = None, None

    cont_sequence = nn.Sequential(*cont_layers)
    return cont_sequence, current_shape, extrema


def translate_simple_layer(layer):
    if layer.__class__ in (nn.ReLU, nn.LeakyReLU, nn.SiLU, nn.GELU):
        return inn.translate_activation(layer)

    elif isinstance(layer, nn.modules.batchnorm._BatchNorm) or isinstance(layer, nn.modules.instancenorm._InstanceNorm):
        return inn.translate_norm(layer)

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

def replace_conv_kernels(network, k_type='mlp'):
    length = len(network)
    for i in range(length):
        m = network[i]
        if isinstance(m, nn.Sequential):
            replace_conv_kernels(m)
        elif hasattr(m, 'sequential'):
            replace_conv_kernels(m.sequential)
        elif isinstance(m, inn.SplineConv):
            network[i] = replace_conv_kernel(m)


def replace_conv_kernel(layer, k_type='mlp'):
    #if k_type
    if isinstance(layer, inn.SplineConv):
        conv = inn.MLPConv(layer.in_channels, layer.out_channels, layer.kernel_size, stride=layer.stride, groups=layer.groups)
        conv.padded_extrema = layer.padded_extrema
        conv.bias = layer.bias
        return conv
    raise NotImplementedError

