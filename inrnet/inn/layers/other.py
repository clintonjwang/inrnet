import torch
nn = torch.nn
F = nn.functional
from inrnet.inn import functional as inrF

class PositionalEncoding(nn.Module):
    def __init__(self, N=4, additive=True): #N*4 channels
        super().__init__()
        self.N = N
        self.additive = additive
    def __repr__(self):
        return f"""PositionalEncoding(N={self.N})"""
    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.set_integrator(inrF.pos_enc, 'PositionalEncoding', layer=self)
        return new_inr
