import torch, pdb
nn = torch.nn

from inrnet import inn

def translate_discrete_layer(layer, img_shape=(224, 224)):
    if isinstance(layer, nn.Conv2d):
        return inn.translate_conv2d(layer, img_shape=img_shape)
    elif isinstance(layer, nn.BatchNorm2d):
        return inn.translate_bn(layer)
    elif isinstance(layer, nn.MaxPool2d):
        return inn.translate_pool(layer)
    elif isinstance(layer, nn.ReLU):
        return inn.translate_activation(layer)
    else:
        raise NotImplementedError(layer.__class__)
