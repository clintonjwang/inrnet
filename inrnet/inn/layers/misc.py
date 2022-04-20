# from functools import partial
# import torch
# nn = torch.nn
# F = nn.functional

# class LayerWrapper(nn.Module):
#     def __init__(self, torch_layer):
#         super().__init__()
#         self.torch_layer = torch_layer
#     def forward(self, vvf):
#         vvf.add_modification(lambda x: self.torch_layer(x))
#         return vvf