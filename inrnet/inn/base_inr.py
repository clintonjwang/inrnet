"""Class for minibatches of INRs"""
from __future__ import annotations
from typing import Callable, Tuple
import torch

from inrnet.inn.point_set import PointValues
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
    def add_layer(self, modification) -> None:
        self.compute_graph.add_layer('modifier', modification)
        # test_output = modification(torch.randn(1,self.channels).cuda()) #.double()
        # self.channels = test_output.size(1)

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
