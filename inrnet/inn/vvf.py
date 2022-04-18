import numpy as np
import torch, operator, copy, pdb
from functools import partial
nn=torch.nn
F=nn.functional

from inrnet import inn

class VectorValuedFunction(nn.Module):
    def __init__(self, function, output_dims, input_dims=2, domain=(-1,1)):
        super().__init__()
        self.function = function
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.domain = domain
        self.modifiers = []
        self.integrator = None
        if not isinstance(domain, tuple):
            raise NotImplementedError("domain must be an n-cube")

    def forward(self, coords):
        return self.evaluator(coords)