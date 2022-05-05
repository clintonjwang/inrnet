from functools import partial
from typing import Union, List, Dict, Any, Optional, cast

import torch
nn = torch.nn
F = nn.functional

def get_vgg11():
    src_model = torchvision.models.vgg11(pretrained=True)
    inr_net = VGG

    layers = []
    for layer in src_model.features:
        layers.append(inn.conversion.translate_discrete_layer(layer))
    nn.Sequential(layers)

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, inr) -> torch.Tensor:
        inr = self.features(inr)
        x = self.avgpool(inr)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x