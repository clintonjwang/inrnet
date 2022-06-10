"""Layer whose output is the sampling density at the same region"""
from inrnet.inn.inr import INRBatch
from inrnet.inn import functional as inrF

import torch
nn = torch.nn
F = nn.functional
from functools import partial

class SampleDensityLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def __repr__(self):
        return f"""PredictLayer()"""
    def forward(self, inr: INRBatch) -> INRBatch:
        new_inr = inr.create_derived_inr()
        new_inr.add_integrator(inrF.change_sample_density)
        return new_inr
