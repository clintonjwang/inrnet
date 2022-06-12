from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from inrnet.inn.inr import INRBatch
from inrnet.inn.point_set import PointValues
import torch
nn = torch.nn
F = nn.functional
from inrnet.inn import functional as inrF

class PointWiseFunction(nn.Module):
    def __init__(self, function, name): #N*4 channels
        super().__init__()
        self.function = function
        self.name = name

    def __str__(self):
        return self.name

    def forward(self, values: PointValues) -> PointValues:
        return self.function(values)

class PositionalEncoding(nn.Module):
    def __init__(self, N=4, additive=True, scale=1.): #N*4 channels
        super().__init__()
        self.N = N
        self.additive = additive
        self.scale = scale
        
    def __str__(self):
        return 'PosEnc'
    def __repr__(self):
        return f"""PositionalEncoding(N={self.N})"""

    def forward(self, inr: INRBatch) -> INRBatch:
        inr.add_integrator(inrF.pos_enc, 'PositionalEncoding', layer=self)
        return inr
