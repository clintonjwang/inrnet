# deprecated

# import numpy as np
# import torch, operator, copy, pdb
# from functools import partial
# nn=torch.nn
# F=nn.functional

# from inrnet import inn

# class VectorValuedFunction(nn.Module):
#     def __init__(self, inr, integrator, output_dims, input_dims=2, domain=(-1,1)):
#         super().__init__()
#         self.inr = inr
#         self.input_dims = input_dims
#         self.output_dims = output_dims
#         self.domain = domain
#         self.integrator = integrator
#         self.modifiers = []
#         if not isinstance(domain, tuple):
#             raise NotImplementedError("domain must be an n-cube")

#     def add_modification(self, modification):
#         self.modifiers.append(modification)
#         test_output = modification(torch.randn(1,self.output_dims).cuda())
#         self.output_dims = test_output.size(1)

#     def forward(self, coords):
#         out = self.integrator(self.inr(coords))
#         for m in self.modifiers:
#             out = m(out)
#         return out