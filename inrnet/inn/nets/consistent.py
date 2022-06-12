from torch import nn
from inrnet import inn
from inrnet.inn.inr import DiscretizedINR
from inrnet.inn.layers.other import FlowLayer
from inrnet.inn.nets.inrnet import INRNet

class ConsistentNet(INRNet):
    def __init__(self, dense_sampler, flow_layers: nn.Module, ratio=8, **kwargs):
        """_summary_

        Args:
            dense_sampler (_type_): full QMC sampler
            layers (nn.Module): _description_
        """        
        super().__init__(sampler=dense_sampler, layers=flow_layers, **kwargs)
        sparse_sampler = dense_sampler
        sparse_sampler['sample points'] = dense_sampler['sample points']//ratio
        
    def forward(self, inr: DiscretizedINR):
        return

    def loss(self, sampler):
        return
