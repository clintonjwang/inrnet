import numpy as np
import torch, operator, copy, pdb
from functools import partial
nn=torch.nn
F=nn.functional
from time import time

from inrnet import util
from inrnet.inn import functional as inrF


class INRBatch(nn.Module):
    def __init__(self, channels, sample_size=256, input_dims=2, domain=(-1,1)):
        super().__init__()
        self.input_dims = input_dims # input coord size
        self.channels = channels # output size
        self.domain = domain
        self.modifiers = [] # in-place/pointwise modifiers
        self.integrator = None # operators requiring integration
        self.sample_size = sample_size
        self.detached = False
        self.grid_mode = False
        self.caching_enabled = True
        if not isinstance(domain, tuple):
            raise NotImplementedError("domain must be an n-cube")

    @property
    def volume(self):
        if isinstance(self.domain, tuple):
            return (self.domain[1] - self.domain[0])**self.input_dims
        else:
            return np.prod([d[1]-d[0] for d in self.domain])
            
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

    def toggle_grid_mode(self, mode=None):
        if mode is None:
            mode = not self.grid_mode
        elif self.grid_mode == mode:
            return
        self.grid_mode = mode
        if hasattr(self.evaluator, 'toggle_grid_mode'):
            self.evaluator.toggle_grid_mode(mode=mode)

    def __neg__(self):
        self.add_modification(lambda x: -x)
        return self

    def __add__(self, other):
        if isinstance(other, INRBatch):
            return SumINR(self, other)
        else:
            return self.create_modified_copy(lambda x: x+other)
    def __iadd__(self, other):
        self.add_modification(lambda x: x+other)
        return self

    def __sub__(self, other):
        if isinstance(other, INRBatch):
            return SumINR(self, -other)
        else:
            return self.create_modified_copy(lambda x: x-other)
    def __isub__(self, other):
        self.add_modification(lambda x: x-other)
        return self

    def __mul__(self, other):
        if isinstance(other, INRBatch):
            return MulINR(self, -other)
        else:
            return self.create_modified_copy(lambda x: x*other)
    def __imul__(self, other):
        self.add_modification(lambda x: x*other)
        return self

    def __truediv__(self, other):
        if isinstance(other, INRBatch):
            return MulINR(self, 1/other)
        else:
            return self.create_modified_copy(lambda x: x/other)
    def __itruediv__(self, other):
        self.add_modification(lambda x: x/other)
        return self
    def __rtruediv__(self, other):
        return self.create_modified_copy(lambda x: other/x)
        # self.add_modification(lambda x: other/x)
        # return self
    
    def cat(self, other):
        if isinstance(other, INRBatch):
            return CatINR(self, other)
        else:
            return self.create_modified_copy(lambda x: torch.cat(x,other))

    def generate_sample_points(self, method="qmc", sample_size=None, dims=None, ordering='c2f'):
        if sample_size is None:
            sample_size = self.sample_size
        if method == "grid" or self.grid_mode:
            if dims is None:
                raise ValueError("declare dims or turn off grid mode")
            return util.meshgrid_coords(*dims, c2f=ordering=='c2f')
        elif method in ("qmc", 'rqmc'):
            return inrF.generate_quasirandom_sequence(d=self.input_dims, n=sample_size,
                bbox=(*self.domain, *self.domain), scramble=(method=='rqmc'))
        else:
            raise NotImplementedError("invalid method: "+method)

    def set_integrator(self, function, name, layer=None, **kwargs):
        self.integrator = inrF.Integrator(function, name, inr=self, layer=layer, **kwargs)

    def add_modification(self, modification):
        self.sampled_coords = torch.empty(0)
        self.modifiers.append(modification)
        # test_output = modification(torch.randn(1,self.channels).cuda()) #.double()
        # self.channels = test_output.size(1)

    def create_modified_copy(self, modification):
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

    def pow(self, n, inplace=False):
        if inplace:
            self.add_modification(lambda x: x.pow(n))
            return self
        else:
            return self.create_modified_copy(lambda x: x.pow(n))

    def sqrt(self, inplace=False):
        if inplace:
            self.add_modification(torch.sqrt)
            return self
        else:
            return self.create_modified_copy(torch.sqrt)

    def log(self, inplace=False):
        if inplace:
            self.add_modification(torch.log)
            return self
        else:
            return self.create_modified_copy(torch.log)

    def sigmoid(self, inplace=False):
        if inplace:
            self.add_modification(torch.sigmoid)
            return self
        else:
            return self.create_modified_copy(torch.sigmoid)

    def softmax(self, inplace=False):
        if inplace:
            self.add_modification(lambda x: torch.softmax(x,dim=-1))
            return self
        else:
            return self.create_modified_copy(lambda x: torch.softmax(x,dim=-1))

    def matmul(self, other, inplace=False):
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
        else:
            return self._forward(coords)

    def _forward(self, coords):
        if hasattr(self, "cached_outputs"):
            return self.cached_outputs

        out = self.evaluator(coords)
        try:
            self.sampled_coords = self.evaluator.sampled_coords
        except AttributeError:
            self.sampled_coords = self.origin.sampled_coords

        if self.integrator is not None:
            out = self.integrator(out)
        for m in self.modifiers:
            out = m(out)
        if self.caching_enabled:
            self.cached_outputs = out
        return out

    def produce_images(self, H,W, dtype=torch.float):
        with torch.no_grad():
            xy_grid = util.meshgrid_coords(H,W)
            output = self.forward(xy_grid)
            output = util.realign_values(output, inr=self)
            output = output.reshape(output.size(0),H,W,-1)
        if dtype == 'numpy':
            return output.squeeze(-1).cpu().float().numpy()
        else:
            return output.permute(0,3,1,2).to(dtype=dtype)



class BlackBoxINR(INRBatch):
    """wrapper for arbitrary INR architectures (SIREN, NeRF, etc.)"""
    def __init__(self, evaluator, channels, **kwargs):
        super().__init__(channels=channels, **kwargs)
        self.evaluator = evaluator

    def __repr__(self):
        return f"""BlackBoxINR(batch_size={len(self.evaluator)}, channels={self.channels}, modifiers={self.modifiers})"""

    def forward(self, coords):
        if hasattr(self, "cached_outputs") and self.sampled_coords.shape == coords.shape and torch.allclose(self.sampled_coords, coords):
            return self.cached_outputs
        self.sampled_coords = coords
        with torch.no_grad():
            out = []
            for inr in self.evaluator:
                out.append(inr(coords))
            out = torch.stack(out, dim=0)
        for m in self.modifiers:
            out = m(out)
        self.cached_outputs = out
        return out

    def forward_with_grad(self, coords):
        self.sampled_coords = coords
        out = []
        for inr in self.evaluator:
            out.append(inr(coords))
        out = torch.stack(out, dim=0)
        for m in self.modifiers:
            out = m(out)
        return out


class CondINR(INRBatch):
    def __init__(self, channels, cond_integrator=False, sample_size=256, input_dims=2, domain=(-1,1)):
        super().__init__(channels, sample_size, input_dims, domain)
        self.cond_integrator = cond_integrator
    def create_derived_inr(self):
        new_inr = CondINR(channels=self.channels, input_dims=self.input_dims, domain=self.domain)
        new_inr.evaluator = self
        return new_inr
    def create_conditional_inr(self):
        new_inr = CondINR(cond_integrator=True, channels=self.channels, input_dims=self.input_dims, domain=self.domain)
        new_inr.evaluator = self
        return new_inr
    def forward(self, coords, condition):
        if self.detached:
            if hasattr(self, "cached_outputs"):
                return self.cached_outputs.detach()
            with torch.no_grad():
                return self._forward(coords, condition)
        else:
            return self._forward(coords, condition)
    def _forward(self, coords, condition):
        if hasattr(self, "cached_outputs"):
            return self.cached_outputs
        if isinstance(self.evaluator, CondINR):
            out = self.evaluator(coords, condition)
        else:
            out = self.evaluator(coords)
        try:
            self.sampled_coords = self.evaluator.sampled_coords
        except AttributeError:
            self.sampled_coords = self.origin.sampled_coords
        if self.integrator is not None:
            if self.cond_integrator:
                out = self.integrator(out, condition)
            else:
                out = self.integrator(out)
        for m in self.modifiers:
            out = m(out)
        if self.caching_enabled:
            self.cached_outputs = out
        return out

def merge_domains(d1, d2):
    return (max(d1[0], d2[0]), min(d1[1], d2[1]))
    
def set_difference(x,y):
    combined = torch.cat((x, y))
    uniques, counts = combined.unique(return_counts=True)
    return uniques[counts == 1]

class MergeINR(INRBatch):
    def __init__(self, inr1, inr2, channels, merge_function):
        domain = merge_domains(inr1.domain, inr2.domain)
        super().__init__(channels=channels, input_dims=inr1.input_dims, domain=domain)
        self.inr1 = inr1
        self.evaluator = self.inr2 = inr2
        self.merge_function = merge_function
        self.interpolator = inrF.interpolate

    def merge_coords(self, values1, values2):
        x = self.inr1.sampled_coords
        y = self.inr2.sampled_coords
        if len(x) == len(y):
            if torch.all(x == y):
                self.sampled_coords = x
                out = self.merge_function(values1, values2)
                return out
            else:
                x_indices = torch.sort((x[:,0]+2)*x.size(0)/2 + x[:,1]).indices
                y_indices = torch.sort((y[:,0]+2)*y.size(0)/2 + y[:,1]).indices
                self.sampled_coords = x[x_indices]
                if torch.allclose(self.sampled_coords, y[y_indices]):
                    return self.merge_function(values1[:,x_indices], values2[:,y_indices])
                else:
                    print('coord_conflict')
                    pdb.set_trace()

        pdb.set_trace()
        coord_diffs = x.unsqueeze(0) - y.unsqueeze(1)
        matches = (coord_diffs.abs().sum(-1) == 0)
        y_indices, x_indices = torch.where(matches)
        X = values1[x_indices]
        Y = values2[y_indices]
        merged_outs = self.merge_function(X,Y)
        extra_x = set_difference(torch.arange(len(x), device=x.device), x_indices)
        extra_x_vals = self.merge_function(values1[extra_x],
            self.interpolator(query_coords=x[extra_x], observed_coords=y, values=values2))
        extra_y = set_difference(torch.arange(len(y), device=x.device), y_indices)
        extra_y_vals = self.merge_function(values2[extra_y],
            self.interpolator(query_coords=y[extra_y], observed_coords=x, values=values1))
        self.sampled_coords = torch.cat((x[x_indices], x[extra_x], y[extra_y]), dim=0)
        
        return torch.cat((merged_outs, values1[extra_x], values2[extra_y]), dim=0)

    def _forward(self, coords):
        if hasattr(self, "cached_outputs") and self.sampled_coords.shape == coords.shape and torch.allclose(self.sampled_coords, coords):
            return self.cached_outputs
        out1 = self.inr1(coords)
        out2 = self.inr2(coords)
        out = self.merge_coords(out1, out2)
        if self.integrator is not None:
            out = self.integrator(out)
        for m in self.modifiers:
            out = m(out)
        self.cached_outputs = out
        return out

    def toggle_grid_mode(self, mode=None):
        if mode is None:
            mode = not self.grid_mode
        self.grid_mode = mode
        self.inr1.toggle_grid_mode(mode=mode)
        self.inr2.toggle_grid_mode(mode=mode)

class SumINR(MergeINR):
    def __init__(self, inr1, inr2):
        super().__init__(inr1, inr2, channels=inr1.channels, merge_function=operator.__add__)
class MulINR(MergeINR):
    def __init__(self, inr1, inr2):
        super().__init__(inr1, inr2, channels=inr1.channels, merge_function=operator.__mul__)
# class MatMulINR(MergeINR):
#     def __init__(self, inr1, inr2):
#         super().__init__(inr1, inr2, channels=inr2.channels, merge_function=torch.matmul)
class CatINR(MergeINR):
    def __init__(self, inr1, inr2):
        super().__init__(inr1, inr2, channels=inr1.channels+inr2.channels,
            merge_function=lambda x,y:torch.cat((x,y),dim=-1))


# class SplitINR(INRBatch):
#     # splits an INR into 2 INRs, one of split_channel and one of c_out - split_channel
#     def __init__(self, inr, split_channel, merge_function):
#         super().__init__(channels=channels, input_dims=inr1.input_dims, domain=domain)
#         self.inr1 = inr1
#         self.channels1 = split_channel
#         self.channels2 = inr1.channels - split_channel
#         raise NotImplementedError('splitinr')

#     def _forward(self, coords):
#         return torch.split(self.inr1(coords))
