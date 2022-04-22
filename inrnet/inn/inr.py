import numpy as np
import torch, operator, copy, pdb
from functools import partial
nn=torch.nn
F=nn.functional

from inrnet import util
from inrnet.inn import functional as inrF

class INR(nn.Module):
    def __init__(self, channels, sample_size=256, input_dims=2, domain=(-1,1)):
        super().__init__()
        self.input_dims = input_dims # input coord size
        self.channels = channels # output size
        self.domain = domain
        self.modifiers = []
        self.integrator = None
        self.sample_size = sample_size
        self.detached = False
        if not isinstance(domain, tuple):
            raise NotImplementedError("domain must be an n-cube")

    @property
    def volume(self):
        if isinstance(self.domain, tuple):
            return (self.domain[1] - self.domain[0])**self.input_dims
        else:
            return np.prod([d[1]-d[0] for d in self.domain])

    def __neg__(self):
        self.add_modification(lambda x: -x)
        return self

    def __add__(self, other):
        if isinstance(other, INR):
            return SumINR(self, other)
        else:
            return self.create_modified_copy(lambda x: x+other)
    def __iadd__(self, other):
        self.add_modification(lambda x: x+other)
        return self

    def __sub__(self, other):
        if isinstance(other, INR):
            return SumINR(self, -other)
        else:
            return self.create_modified_copy(lambda x: x-other)
    def __isub__(self, other):
        self.add_modification(lambda x: x-other)
        return self

    def __mul__(self, other):
        if isinstance(other, INR):
            return MulINR(self, -other)
        else:
            return self.create_modified_copy(lambda x: x*other)
    def __imul__(self, other):
        self.add_modification(lambda x: x*other)
        return self

    def __truediv__(self, other):
        if isinstance(other, INR):
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
    
    def generate_sample_points(self, method="qmc", sample_size=None, dims=None):
        if sample_size is None:
            sample_size = self.sample_size
        if method == "qmc":
            return inrF.generate_quasirandom_sequence(d=self.input_dims, n=sample_size) * \
                    (self.domain[1]-self.domain[0]) - self.domain[0]
        elif method == "grid":
            return util.meshgrid_coords(*dims)
        else:
            raise NotImplementedError("invalid method: "+method)

    def add_modification(self, modification):
        self.modifiers.append(modification)
        test_output = modification(torch.randn(1,self.channels).cuda())
        self.channels = test_output.size(1)

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
        if isinstance(other, INR):
            return MatMulINR(self, other)
        elif inplace:
            self.add_modification(lambda x: torch.matmul(x,other))
            return self
        else:
            return self.create_modified_copy(lambda x: torch.matmul(x,other))

    # def cat(self, other):
    #     return CatINR(self, other)

    def create_modified_copy(self, modification):
        new_inr = copy.copy(self)
        new_inr.modifiers = self.modifiers.copy()
        new_inr.add_modification(modification)
        return new_inr

    def create_derived_inr(self):
        new_inr = INR(channels=self.channels, input_dims=self.input_dims, domain=self.domain)
        new_inr.evaluator = self
        return new_inr

    # def create_VVF(self, integrator):
    #     VVF = vvf.VectorValuedFunction(inr=self, integrator=integrator,
    #         output_dims=self.channels, input_dims=self.input_dims, domain=self.domain)
    #     return VVF

    def detach(self):
        self.detached = True
        return self

    def produce_image(self, H,W, split=None):
        with torch.no_grad():
            if split == None:
                xy_grid = util.meshgrid_coords(H,W)
                output = self.forward(xy_grid)
                output = util.realign_values(output, coords_gt=xy_grid, inr=self)
                return output.reshape(H,W,-1).squeeze(-1).float().cpu().numpy()
            else:
                outs = []
                xy_grids = util.meshgrid_split_coords(H,W,split=split, device="cpu")
                for xy_grid in xy_grids:
                    output = self.forward(xy_grid.cuda())
                    outs.append(util.realign_values(output, coords_gt=xy_grid.cuda(), inr=self, split=16).cpu())
                    torch.cuda.empty_cache()
                xy_grid = util.meshgrid_coords(H,W)
                output = util.realign_values(torch.cat(outs, dim=0).cuda(), coords_gt=xy_grid,
                            coords_out=torch.cat(xy_grids, dim=0).cuda(), split=32)
                return output.reshape(H,W,-1).squeeze(-1).cpu().float().numpy()

    def forward(self, coords):
        if self.detached:
            with torch.no_grad():
                return self._forward(coords)
        else:
            return self._forward(coords)

    def _forward(self, coords):
        out = self.evaluator(coords)
        self.sampled_coords = self.evaluator.sampled_coords
        if self.integrator is not None:
            out = self.integrator(out)
        for m in self.modifiers:
            out = m(out)
        return out


class BlackBoxINR(INR):
    # wrapper for arbitrary INR architectures (SIREN, NeRF, etc.)
    def __init__(self, evaluator, channels, device="cuda", dtype=torch.float, **kwargs):
        super().__init__(channels=channels, **kwargs)
        self.evaluator = evaluator.to(device=device, dtype=dtype)

    def add_modification(self, modification):
        super().add_modification(modification)
        self.sampled_coords = torch.empty(0)

    def create_modified_copy(self, modification):
        self.sampled_coords = torch.empty(0)
        new_inr = super().create_modified_copy(modification)
        return new_inr

    def forward(self, coords):
        if hasattr(self, "cached_outputs") and self.sampled_coords.shape == coords.shape and torch.allclose(self.sampled_coords, coords):
            return self.cached_outputs

        self.sampled_coords = coords
        with torch.no_grad():
            out = self.evaluator(coords)
        for m in self.modifiers:
            out = m(out)
        self.cached_outputs = out
        return out

    def forward_with_grad(self, coords):
        self.sampled_coords = coords
        out = self.evaluator(coords)
        for m in self.modifiers:
            out = m(out)
        return out



def merge_domains(d1, d2):
    return (max(d1[0], d2[0]), min(d1[1], d2[1]))
    
def set_difference(x,y):
    combined = torch.cat((x, y))
    uniques, counts = combined.unique(return_counts=True)
    return uniques[counts == 1]

class MergeINR(INR):
    def __init__(self, inr1, inr2, channels, merge_function):
        domain = merge_domains(inr1.domain, inr2.domain)
        super().__init__(channels=channels, input_dims=inr1.input_dims, domain=domain)
        self.inr1 = inr1
        self.inr2 = inr2
        self.merge_function = merge_function
        self.interpolator = inrF.interpolate

    def merge_coords(self, values1, values2):
        x = self.inr1.sampled_coords
        y = self.inr2.sampled_coords
        if len(x) == len(y) and torch.all(x == y):
            self.sampled_coords = x
            return self.merge_function(values1, values2)

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

    def forward(self, coords):
        out = self.merge_coords(self.inr1(coords), self.inr2(coords))
        if self.integrator is not None:
            out = self.integrator(out)
        for m in self.modifiers:
            out = m(out)
        return out


class SumINR(MergeINR):
    def __init__(self, inr1, inr2):
        super().__init__(inr1, inr2, channels=inr1.channels, merge_function=operator.__add__)
class MulINR(MergeINR):
    def __init__(self, inr1, inr2):
        super().__init__(inr1, inr2, channels=inr1.channels, merge_function=operator.__mul__)
class MatMulINR(MergeINR):
    def __init__(self, inr1, inr2):
        super().__init__(inr1, inr2, channels=inr2.channels, merge_function=torch.matmul)
class CatINR(MergeINR):
    def __init__(self, inr1, inr2):
        super().__init__(inr1, inr2, channels=inr1.channels+inr2.channels,
            merge_function=lambda x,y:torch.cat((x,y),dim=-1))


class SplitINR(INR):
    # splits an INR into 2 INRs, one of split_channel and one of c_out - split_channel
    def __init__(self, inr, split_channel, merge_function):
        super().__init__(channels=channels, input_dims=inr1.input_dims, domain=domain)
        self.inr1 = inr1
        self.channels1 = split_channel
        self.channels2 = inr1.channels - split_channel
        raise NotImplementedError

    def forward(self, coords):
        raise NotImplementedError
        return torch.split(self.inr1(coords))

