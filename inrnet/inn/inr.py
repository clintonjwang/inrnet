import numpy as np
import torch, operator, copy, pdb
from functools import partial
nn=torch.nn
F=nn.functional

from inrnet.inn import vvf

class INR(nn.Module):
    def __init__(self, channels, input_dims=2, domain=(-1,1)):
        super().__init__()
        self.input_dims = input_dims # input coord size
        self.channels = channels # output size
        self.domain = domain
        self.modifiers = []
        self.integrator = None
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
    
    @staticmethod
    def merge_domains(d1, d2):
        return (max(d1[0], d2[0]), min(d1[1], d2[1]))

    def add_modification(self, modification):
        self.modifiers.append(modification)
        # if hasattr(self, "sampled_coords"):
        #     new_vals = modification(self.sampled_coords[:,self.input_dims:])
        #     self.channels = new_vals.size(1)
        #     self.sampled_coords = torch.cat((self.sampled_coords[:,:self.input_dims], new_vals), dim=1)
        # else:
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
            self.add_modification(lambda x: x.sqrt())
            return self
        else:
            return self.create_modified_copy(lambda x: x.sqrt())

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

    def cat(self, other):
        return CatINR(self, other)

    def create_modified_copy(self, modification):
        new_inr = copy.copy(self)
        new_inr.modifiers = self.modifiers.copy()
        new_inr.add_modification(modification)
        return new_inr

    def create_derived_inr(self):
        new_inr = INR(channels=self.channels, input_dims=self.input_dims, domain=self.domain)
        new_inr.evaluator = self
        return new_inr

    def create_VVF(self, function):
        VVF = vvf.VectorValuedFunction(evaluator=self, modifiers=[function],
            output_dims=self.channels, input_dims=self.input_dims, domain=self.domain)
        return VVF

    # def mean(self, values):
    #     return self.sampled_coords[:, self.input_dims:].mean(0)

    def forward(self, coords):
        out = self.evaluator(coords)
        for m in self.modifiers:
            out = m(out)
        if self.integrator is not None:
            out = self.integrator(coords, out)
        # if not self.training:
        #     self.sampled_coords = torch.cat((coords, out), dim=1)
        return out


class BlackBoxINR(INR):
    # wrapper for arbitrary INR architectures (SIREN, NeRF, etc.)
    def __init__(self, evaluator, channels, input_dims=2, domain=(-1,1), device="cuda", dtype=torch.float):
        super().__init__(channels=channels, input_dims=input_dims, domain=domain)
        self.evaluator = evaluator.to(device=device, dtype=dtype)
        # init_coords = torch.tensor(inrF.generate_quasirandom_sequence(d=input_dims, n=128) * \
        #                     (domain[1]-domain[0]) - domain[0]).to(device=device, dtype=dtype)
        # vals = self.forward(init_coords)
        # self.sampled_coords = torch.cat((init_coords, vals), dim=-1)


    def forward(self, coords):
        with torch.no_grad():
            out = self.evaluator(coords)
        for m in self.modifiers:
            out = m(out)
        if self.integrator is not None:
            out = self.integrator(coords, out)
        # if not self.training:
        #     self.sampled_coords = torch.cat((coords, out), dim=1)
        return out

    def forward_with_grad(self, coords):
        out = self.evaluator(coords)
        for m in self.modifiers:
            out = m(out)
        if self.integrator is not None:
            out = self.integrator(coords, out)
        return out



class MergeINR(INR):
    def __init__(self, inr1, inr2, channels, merge_function):
        domain = INR.merge_domains(inr1.domain, inr2.domain)
        super().__init__(channels=channels, input_dims=inr1.input_dims, domain=domain)
        self.inr1 = inr1
        self.inr2 = inr2
        self.merge_function = merge_function
        # self.merge_coords()

    def merge_coords(self):
        if not hasattr(self.inr1, "sampled_coords"):
            if not hasattr(self.inr2, "sampled_coords"):
                return
            else:
                self.sampled_coords = self.inr2.sampled_coords
        elif not hasattr(self.inr2, "sampled_coords"):
            self.sampled_coords = self.inr1.sampled_coords
        else:
            x = self.inr1.sampled_coords
            y = self.inr2.sampled_coords
            # coord_diffs = torch.einsum('iv,jv->ijv', x[:,:self.input_dims], -y[:,:self.input_dims])
            coord_diffs = x[:,:self.input_dims].unsqueeze(0) - y[:,:self.input_dims].unsqueeze(1)
            matches = (coord_diffs.abs().sum(-1) == 0)
            y_indices, x_indices = torch.where(matches)
            X = x[x_indices,self.input_dims:]
            Y = y[y_indices,self.input_dims:]
            self.sampled_coords = torch.cat([x[x_indices,:self.input_dims], self.merge_function(X,Y)], dim=-1)
            # for coord, val1 in self.inr1.sampled_coords.items():
            #     if coord in self.inr2.sampled_coords:
            #         val2 = inr2.sampled_coords
            #         self.sampled_coords[coord] = self.merge_function(val1, val2)

    def forward(self, coords):
        out = self.merge_function(self.inr1(coords), self.inr2(coords))
        for m in self.modifiers:
            out = m(out)
        if self.integrator is not None:
            out = self.integrator(coords, out)
        # if not self.training:
        #     self.sampled_coords = torch.cat((coords, out), dim=1)
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
        domain = INR.merge_domains(inr1.domain, inr2.domain)
        super().__init__(channels=channels, input_dims=inr1.input_dims, domain=domain)
        self.inr1 = inr1
        self.channels1 = split_channel
        self.channels2 = inr1.channels - split_channel
        raise NotImplementedError

    def forward(self, coords):
        raise NotImplementedError
        return torch.split(self.inr1(coords))
