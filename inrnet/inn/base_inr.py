"""Class for minibatches of INRs"""
from __future__ import annotations
import operator
import pdb
from typing import Callable, Tuple
import torch
from inrnet.inn.nets.inrnet import INRNet

from inrnet.inn.point_set import PointSet, PointValues
nn=torch.nn
F=nn.functional

from inrnet import util

class INRBase(nn.Module):
    """Standard INR minibatch"""
    def __init__(self, channels: int,
        input_dims: int=2, domain: Tuple[int]=(-1,1),
        sample_mode='qmc', device='cuda'):
        """
        Args:
            channels (int): output size
            input_dims (int, optional): input coord size. Defaults to 2.
            domain (Tuple[int], optional): _description_. Defaults to (-1,1).
            device (str, optional): Defaults to 'cuda'.
        """
        super().__init__()
        self.input_dims = input_dims
        self.channels = channels
        self.domain = domain
        self.modifiers = [] # in-place/pointwise modifiers
        self.integrator = None # operators requiring integration
        self.detached = False
        self.sample_mode = sample_mode
        self.caching_enabled = True
        self.device = device
        self.compute_graph = INRNet()
        if not isinstance(domain, tuple):
            raise NotImplementedError("domain must be an n-cube")

    def parent(self, n=1):
        ev = self
        for _ in range(n):
            if hasattr(ev, 'evaluator'):
                ev = ev.evaluator
            else:
                print("no parent")
                return ev
        return ev

    def evaluator_iter(self):
        ev = self
        while True:
            if hasattr(ev, 'evaluator'):
                ev = ev.evaluator
                yield ev
            else:
                return

    def change_sample_mode(self, mode='grid'):
        if self.sample_mode == mode:
            return
        self.sample_mode = mode
        if hasattr(self.evaluator, 'change_sample_mode'):
            self.evaluator.change_sample_mode(mode=mode)

    def __neg__(self):
        self.add_modification(lambda x: -x)
        return self

    def add_integrator(self, function, name, layer=None, **kwargs):
        self.compute_graph.add_node('integrator', Integrator(function, name, layer=layer, **kwargs))

    def add_modification(self, modification) -> None:
        self.compute_graph.add_node('modifier', modification)
        # test_output = modification(torch.randn(1,self.channels).cuda()) #.double()
        # self.channels = test_output.size(1)

    def create_modified_copy(self, modification: Callable) -> INRBatch:
        new_inr = self.create_derived_inr()
        new_inr.add_modification(modification)
        return new_inr
    def create_derived_inr(self):
        new_inr = INRBatch(channels=self.channels, input_dims=self.input_dims, domain=self.domain)
        new_inr.evaluator = self
        return new_inr
    def create_conditional_inr(self):
        new_inr = CondINR(cond_integrator=True, channels=self.channels, input_dims=self.input_dims, domain=self.domain)
        new_inr.evaluator = self
        return new_inr

    def pow(self, n, inplace=False) -> INRBatch:
        if inplace:
            self.add_modification(lambda x: x.pow(n))
            return self
        else:
            return self.create_modified_copy(lambda x: x.pow(n))

    def sqrt(self, inplace=False) -> INRBatch:
        if inplace:
            self.add_modification(torch.sqrt)
            return self
        return self.create_modified_copy(torch.sqrt)

    def log(self, inplace=False) -> INRBatch:
        if inplace:
            self.add_modification(torch.log)
            return self
        return self.create_modified_copy(torch.log)

    def sigmoid(self, inplace=False) -> INRBatch:
        if inplace:
            self.add_modification(torch.sigmoid)
            return self
        return self.create_modified_copy(torch.sigmoid)

    def softmax(self, inplace=False) -> INRBatch:
        if inplace:
            self.add_modification(lambda x: torch.softmax(x,dim=-1))
            return self
        return self.create_modified_copy(lambda x: torch.softmax(x,dim=-1))

    def matmul(self, other, inplace=False) -> INRBatch:
        if isinstance(other, INRBatch):
            return MatMulINR(self, other)
        elif inplace:
            self.add_modification(lambda x: torch.matmul(x,other))
            return self
        else:
            return self.create_modified_copy(lambda x: torch.matmul(x,other))

    def __repr__(self):
        ret = repr(self.evaluator)
        ret += f"\n-> C={self.channels}"
        if self.integrator is not None:
            ret += f", integrator={repr(self.integrator)}"
        if len(self.modifiers) > 0:
            ret += f", modifiers={self.modifiers}"""
        return ret

    def detach(self):
        self.detached = True
        return self

    def forward(self, coords):
        if self.detached:
            if hasattr(self, "cached_outputs"):
                return self.cached_outputs.detach()
            with torch.no_grad():
                return self._forward(coords)
        return self._forward(coords)

    def _forward(self, coords):
        if hasattr(self, "cached_outputs"):
            return self.cached_outputs

        out = self.evaluator(coords)
        # try:
        self.sampled_coords = self.evaluator.sampled_coords
        if hasattr(self.evaluator, 'dropped_coords'):
            self.dropped_coords = self.evaluator.dropped_coords
        # except AttributeError:
        #     self.sampled_coords = self.origin.sampled_coords
        
        if self.integrator is not None:
            out = self.integrator(out)
        for m in self.modifiers:
            out = m(out)
        if self.caching_enabled:
            self.cached_outputs = out
        return out

    def produce_images(self, H,W, dtype=torch.float):
        with torch.no_grad():
            xy_grid = util.meshgrid_coords(H,W, device=self.device)
            output = self.forward(xy_grid)
            output = util.realign_values(output, inr=self)
            output = output.reshape(output.size(0),H,W,-1)
        if dtype == 'numpy':
            return output.squeeze(-1).cpu().float().numpy()
        else:
            return output.permute(0,3,1,2).to(dtype=dtype).as_subclass(PointValues)
