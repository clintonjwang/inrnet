import torch, pdb
nn = torch.nn

from inrnet import inn

def translate_discrete_layer(layer, input_shape=(224, 224), **kwargs):
    if isinstance(layer, nn.Conv2d):
        return inn.translate_conv2d(layer, input_shape=input_shape, **kwargs)
    elif isinstance(layer, nn.BatchNorm2d):
        return inn.translate_bn(layer, **kwargs)
    elif isinstance(layer, nn.MaxPool2d):
        return inn.translate_pool(layer, input_shape=input_shape, **kwargs)
    elif isinstance(layer, nn.ReLU):
        return inn.translate_activation(layer, **kwargs)
    else:
        raise NotImplementedError(layer.__class__)
