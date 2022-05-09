import torch, pdb
nn = torch.nn

from inrnet import inn

def freeze_layer_types(network, classes=(inn.ChannelMixer, inn.ChannelNorm)):
    for m in network:
        if hasattr(m, '__iter__'):
            freeze_layer_types(m, classes)
        elif m.__class__ in classes:
            for param in m.parameters():
                param.requires_grad = False

def unfreeze_layer_types(network, classes=(inn.ChannelMixer, inn.ChannelNorm)):
    for m in network:
        if hasattr(m, '__iter__'):
            unfreeze_layer_types(m, classes)
        elif m.__class__ in classes:
            for param in m.parameters():
                param.requires_grad = True

def replace_conv_kernels(network, k_type='mlp', k_ratio=1.):
    length = len(network)
    for i in range(length):
        m = network[i]
        if hasattr(m, '__getitem__'):
            replace_conv_kernels(m, k_ratio=k_ratio)
        elif isinstance(m, inn.SplineConv):
            network[i] = replace_conv_kernel(m, k_ratio=k_ratio)

def replace_conv_kernel(layer, k_type='mlp', k_ratio=1.):
    #if k_type
    if isinstance(layer, inn.SplineConv):
        conv = inn.MLPConv(layer.in_channels, layer.out_channels, [k*k_ratio for k in layer.kernel_size],
            stride=layer.stride, groups=layer.groups)
        # conv.padded_extrema = layer.padded_extrema
        conv.bias = layer.bias
        return conv
    raise NotImplementedError

