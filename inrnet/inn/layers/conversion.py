import torch, pdb
nn = torch.nn

from inrnet import inn

def translate_discrete_layer(layer, img_shape=(224, 224), **kwargs):
    if isinstance(layer, nn.Conv2d):
        return inn.translate_conv2d(layer, img_shape=img_shape, **kwargs)
    elif isinstance(layer, nn.BatchNorm2d):
        return inn.translate_bn(layer, **kwargs)
    elif isinstance(layer, nn.MaxPool2d):
        return inn.translate_pool(layer, img_shape=img_shape, **kwargs)
    elif isinstance(layer, nn.ReLU):
        return inn.translate_activation(layer, **kwargs)
    else:
        raise NotImplementedError(layer.__class__)
