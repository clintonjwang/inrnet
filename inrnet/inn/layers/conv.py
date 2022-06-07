import torch, pdb, math
import numpy as np
from functools import partial
nn = torch.nn
F = nn.functional

from inrnet.inn import qmc, functional as inrF, polynomials
from scipy.interpolate import RectBivariateSpline as Spline2D

def get_kernel_size(input_shape, extrema=((-1,1),(-1,1)), k=3):
    h,w = input_shape
    if isinstance(k, int):
        k = (k,k)
    extrema_dists = extrema[0][1] - extrema[0][0], extrema[1][1] - extrema[1][0]
    if h == 1 or w == 1:
        raise ValueError('input shape too small')
    spacing = extrema_dists[0] / (h-1), extrema_dists[1] / (w-1)
    return k[0] * spacing[0], k[1] * spacing[1]

def translate_conv2d(conv2d, input_shape, extrema=((-1,1),(-1,1)), zero_at_bounds=False, smoothing=.05, **kwargs): #h,w
    # offset/grow so that the conv kernel goes a half pixel past the boundary
    h,w = input_shape # shape of input features/image
    out_, in_, k1, k2 = conv2d.weight.shape
    extrema_dists = extrema[0][1] - extrema[0][0], extrema[1][1] - extrema[1][0]
    if h == 1 or w == 1:
        raise ValueError('input shape too small')
    spacing = extrema_dists[0] / (h-1), extrema_dists[1] / (w-1)
    K = k1 * spacing[0], k2 * spacing[1]
    if zero_at_bounds:
        order = min(4,k1)
    else:
        order = min(3,k1-1)

    if k1 > 3:
        smoothing = 0.
        # cannot handle different knot positions per channel
    if k1 % 2 == k2 % 2 == 0:
        shift = spacing[0]/2, spacing[1]/2
    else:
        shift = 0,0

    # if conv2d.padding != ((k1+1)//2-1, (k2+1)//2-1):
    #     raise NotImplementedError("padding")
    padded_extrema=((extrema[0][0]-spacing[0]/2, extrema[0][1]+spacing[0]/2),
            (extrema[1][0]-spacing[1]/2, extrema[1][1]+spacing[1]/2))
    if conv2d.stride in [1,(1,1)]:
        down_ratio = 1.
        out_shape = input_shape
    elif conv2d.stride == (2,2):
        down_ratio = 1/(conv2d.stride[0]*conv2d.stride[1])
        out_shape = (input_shape[0]//2, input_shape[1]//2)
        extrema = ((extrema[0][0], extrema[0][1]-spacing[0]),
            (extrema[1][0], extrema[1][1]-spacing[1]))
    else:
        raise NotImplementedError("down_ratio")

    bias = conv2d.bias is not None
    layer = SplineConv(in_*conv2d.groups, out_, order=order, smoothing=smoothing,
        init_weights=conv2d.weight.detach().cpu().numpy()*k1*k2,
        # scale up weights since we divide by the number of grid points
        groups=conv2d.groups, shift=shift,
        padded_extrema=padded_extrema, zero_at_bounds=zero_at_bounds,
        # N_bins=0,
        N_bins=2**math.ceil(math.log2(k1*k2)+3), #4
        kernel_size=K, down_ratio=down_ratio, bias=bias, **kwargs)
    if bias:
        layer.bias.data = conv2d.bias.data

    return layer, out_shape, extrema



class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, input_dims=2, down_ratio=1., groups=1, bias=False, dtype=torch.float):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_dims = input_dims
        self.down_ratio = down_ratio
        self.groups = groups
        self.group_size = self.in_channels // self.groups
        self.dtype = dtype
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, dtype=dtype))
        else:
            self.bias = None

    def kernel_intersection_ratio(self, query_coords):
        if not hasattr(self, 'padded_extrema'):
            return
        dist_to_boundary = (query_coords.unsqueeze(1) - self.padded_extrema.T.unsqueeze(0)).abs().amin(dim=1)
        k = self.kernel_size[0]/2, self.kernel_size[1]/2
        padding_ratio = (self.kernel_size[0] - F.relu(k[0] - dist_to_boundary[:,0])) * (
            self.kernel_size[1] - F.relu(k[1] - dist_to_boundary[:,1])) / self.kernel_size[0] / self.kernel_size[1]
        return padding_ratio


def fit_spline(values, K, order=3, smoothing=0, center=(0,0), dtype=torch.float):
    # K = dims of the entire B spline surface
    h,w = values.shape
    bbox = (-K[0]/2+center[0], K[0]/2+center[0], -K[1]/2+center[1], K[1]/2+center[1])
    x,y = (np.linspace(bbox[0]/h*(h-1), bbox[1]/h*(h-1), h),
           np.linspace(bbox[2]/w*(w-1), bbox[3]/w*(w-1), w))

    bs = Spline2D(x,y, values, bbox=bbox, kx=order,ky=order, s=smoothing)
    tx,ty,c = [torch.tensor(z).to(dtype=dtype) for z in bs.tck]
    h=tx.size(0)-order-1
    w=ty.size(0)-order-1
    c=c.reshape(h,w)
    return tx,ty,c


class SplineConv(Conv):
    def __init__(self, in_channels, out_channels, kernel_size, init_weights, order=2, down_ratio=1.,
            input_dims=2, N_bins=0, groups=1, zero_at_bounds=False,
            padded_extrema=None, bias=False, smoothing=0., shift=(0,0),
            dtype=torch.float):
        super().__init__(in_channels, out_channels, input_dims=input_dims,
            down_ratio=down_ratio, bias=bias, groups=groups, dtype=dtype)
        self.N_bins = N_bins
        if not hasattr(kernel_size, '__iter__'):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = K = kernel_size
        if padded_extrema is not None:
            self.register_buffer("padded_extrema", torch.as_tensor(padded_extrema, dtype=dtype))
        self.register_buffer('shift', torch.tensor(shift, dtype=dtype))
        self.diffs_in_support = lambda diffs: (diffs[...,0].abs() < self.kernel_size[0]/2) * (
                        diffs[...,1].abs() < self.kernel_size[1]/2)

        # fit pretrained kernel with b-spline
        h,w = init_weights.shape[2:]
        bbox = (-K[0]/2, K[0]/2, -K[1]/2, K[1]/2)
        x,y = (np.linspace(bbox[0]/h*(h-1), bbox[1]/h*(h-1), h),
               np.linspace(bbox[2]/w*(w-1), bbox[3]/w*(w-1), w))
        if zero_at_bounds:
            x = (bbox[0], *x, bbox[1])
            y = (bbox[2], *y, bbox[3])
            init_weights = np.pad(init_weights,((0,0),(0,0),(1,1),(1,1)))
            # init_weights = F.pad(init_weights,(1,1,1,1))

        self.order = order
        C = []
        for i in range(self.group_size):
            C.append([])
            for o in range(out_channels):
                bs = Spline2D(x,y, init_weights[o,i], bbox=bbox, kx=order,ky=order, s=smoothing)
                tx,ty,c = [torch.tensor(z).to(dtype=dtype) for z in bs.tck]
                h=tx.size(0)-order-1
                w=ty.size(0)-order-1
                C[-1].append(c.reshape(h,w))
            C[-1] = torch.stack(C[-1],dim=0)

        self.C = nn.Parameter(torch.stack(C, dim=1))
        self.register_buffer("grid_points", torch.as_tensor(
            np.dstack(np.meshgrid(x,y)).reshape(-1,2), dtype=dtype))
        if N_bins > 0:
            self.register_buffer("sample_points", qmc.generate_quasirandom_sequence(n=N_bins,
                d=input_dims, bbox=bbox, dtype=dtype))
        self.register_buffer("Tx", tx)
        self.register_buffer("Ty", ty)
        # self.Tx = nn.Parameter(tx)
        # self.Ty = nn.Parameter(ty)

    def __repr__(self):
        return f"""SplineConv(in_channels={self.in_channels}, out_channels={
        self.out_channels}, kernel_size={np.round(self.kernel_size, decimals=3)}, bias={self.bias is not None})"""

    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.set_integrator(inrF.conv, 'SplineConv', layer=self)
        new_inr.channels = self.out_channels
        return new_inr

    def interpolate_weights(self, xy):
        w_oi = []
        X = xy[:,0].unsqueeze(1)
        Y = xy[:,1].unsqueeze(1)
        px = py = self.order

        values, kx = (self.Tx<=X).min(dim=-1)
        values, ky = (self.Ty<=Y).min(dim=-1)
        kx -= 1
        ky -= 1
        kx[values] = self.Tx.size(-1)-px-2
        ky[values] = self.Ty.size(-1)-py-2

        in_, out_ = self.group_size, self.out_channels
        Dim = in_*out_
        Ctrl = self.C.view(Dim, *self.C.shape[-2:])
        for z in range(X.size(0)):
            D = Ctrl[:, kx[z]-px : kx[z]+1, ky[z]-py : ky[z]+1].clone()

            for r in range(1, px + 1):
                try:
                    alphax = (X[z,0] - self.Tx[kx[z]-px+1:kx[z]+1]) / (
                        self.Tx[2+kx[z]-r:2+kx[z]-r+px] - self.Tx[kx[z]-px+1:kx[z]+1])
                except RuntimeError:
                    print("input off the grid")
                    pdb.set_trace()
                for j in range(px, r - 1, -1):
                    D[:,j] = (1-alphax[j-1]) * D[:,j-1] + alphax[j-1] * D[:,j].clone()

            for r in range(1, py + 1):
                alphay = (Y[z,0] - self.Ty[ky[z]-py+1:ky[z]+1]) / (
                    self.Ty[2+ky[z]-r:2+ky[z]-r+py] - self.Ty[ky[z]-py+1:ky[z]+1])
                for j in range(py, r-1, -1):
                    D[:,px,j] = (1-alphay[j-1]) * D[:,px,j-1].clone() + alphay[j-1] * D[:,px,j].clone()
            
            w_oi.append(D[:,px,py])

        return torch.stack(w_oi).view(xy.size(0), self.out_channels, self.group_size)


class MLPConv(Conv):
    def __init__(self, in_channels, out_channels, kernel_size, mid_ch=(32,32), down_ratio=1.,
            input_dims=2, groups=1, padded_extrema=None, bias=False,
            mlp_type='standard', scale1=None, scale2=1,
            dtype=torch.float, N_bins=64):
        super().__init__(in_channels, out_channels, input_dims=input_dims,
            down_ratio=down_ratio, bias=bias, groups=groups, dtype=dtype)
        if not hasattr(kernel_size, '__iter__'):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = K = kernel_size
        # if N_bins is None:
        #     self.N_bins = 2**math.ceil(math.log2(K[0])+12) #2**math.ceil(math.log2(1/K[0]/K[1])+4)
        # else:
        self.N_bins = N_bins
        self.diffs_in_support = lambda diffs: (diffs[...,0].abs() < self.kernel_size[0]/2) * (
                        diffs[...,1].abs() < self.kernel_size[1]/2)

        bbox = (-K[0]/2, K[0]/2, -K[1]/2, K[1]/2)
        if self.N_bins > 0:
            self.register_buffer("sample_points", qmc.generate_quasirandom_sequence(n=self.N_bins,
                d=input_dims, bbox=bbox, dtype=dtype))
            
        if scale1 is None:
            scale1 = (.5/K[0], .5/K[1])
        self.register_buffer("scale1", torch.as_tensor(scale1, dtype=dtype))
        # self.scale2 = scale2
        self.scale2 = scale2
        self.mlp_type = mlp_type
        
        if padded_extrema is not None:
            self.register_buffer("padded_extrema", torch.as_tensor(padded_extrema, dtype=dtype))
        if isinstance(mid_ch, int):
            mid_ch = [mid_ch]

        # layers = [nn.Linear(input_dims, mid_ch[0]), nn.LeakyReLU(inplace=True)]
        # for ix in range(1,len(mid_ch)):
        #     layers += [nn.Linear(mid_ch[ix-1], mid_ch[ix]), nn.LeakyReLU(inplace=True)]
        # self.kernel = nn.Sequential(*layers, nn.Linear(mid_ch[-1], out_channels * self.group_size))

        self.first = nn.Linear(input_dims, mid_ch[0])
        self.first.weight.data.uniform_(-1/input_dims, 1/input_dims)
        layers = []
        for ix in range(1,len(mid_ch)):
            layers += [nn.Linear(mid_ch[ix-1], mid_ch[ix]), nn.ReLU(inplace=True)]
        self.kernel = nn.Sequential(*layers, nn.Linear(mid_ch[-1], out_channels * self.group_size))

        for k in range(0,len(self.kernel),2):
            nn.init.kaiming_uniform_(self.kernel[k].weight)
            self.kernel[k].bias.data.zero_()

    def __repr__(self):
        return f"""MLPConv(in_channels={self.in_channels}, out_channels={
        self.out_channels}, kernel_size={np.round(self.kernel_size, decimals=3)}, bias={self.bias is not None})"""

    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.set_integrator(inrF.conv, 'MLPConv', layer=self)
        new_inr.channels = self.out_channels
        return new_inr

    def interpolate_weights(self, xy):
        # if self.mlp_type == 'siren':
        #     return self.kernel(torch.sin(self.first(xy*self.scale1) * self.scale3)).reshape(
        #         xy.size(0), self.out_channels, self.group_size) * self.scale2
        # else:
        return self.kernel(self.first(xy * self.scale1)).reshape(
            xy.size(0), self.out_channels, self.group_size) * self.scale2



class BallConv(Conv):
    def __init__(self, in_channels, out_channels, radius, down_ratio=1., p_norm="inf",
            input_dims=2, N_bins=16, groups=1, bias=False,
            parameterization="polynomial", padding_mode="cutoff",
            order=3, dropout=0.):
        super().__init__(in_channels, out_channels, input_dims=input_dims,
            down_ratio=down_ratio, bias=bias, groups=groups)
        self.radius = radius
        self.dropout = dropout
        self.N_bins = N_bins
        if p_norm == "inf":
            p_norm = torch.inf
        self.norm = partial(torch.linalg.norm, ord=p_norm, dim=-1)
        self.diffs_in_support = lambda diffs: self.norm(diffs) < self.radius

        if input_dims == 2 and parameterization == "polynomial":
            if p_norm == 2:
                Kernel = polynomials.ZernikeKernel
            elif p_norm == torch.inf:
                Kernel = polynomials.LegendreFilter

            if groups == out_channels and groups == in_channels:
                self.weight = Kernel(in_channels=1,
                    out_channels=out_channels, radius=radius, order=order).cuda()
                raise NotImplementedError("TODO: conv groups")
            else:
                self.weight = Kernel(in_channels=in_channels, out_channels=out_channels,
                    radius=radius, order=order).cuda()
        else:
            if parameterization == "polynomial":
                raise NotImplementedError("TODO: 3D polynomial basis")

            self.weight = nn.Sequential(nn.Linear(input_dims,6), nn.ReLU(inplace=True),
                nn.Linear(6,in_channels*out_channels), Reshape(in_channels,out_channels))
            if groups != 1:
                raise NotImplementedError("TODO: conv groups")
        
        if p_norm not in [2, torch.inf]:
            raise NotImplementedError(f"unsupported norm {p_norm}")
        if parameterization not in ["polynomial", "mlp"]:
            raise NotImplementedError(f"unsupported parameterization {parameterization}")
        if padding_mode not in ["cutoff"]:#, "zeros", "shrink domain", "evaluate"]:
            # cutoff: at each point, evaluate the integral under B intersect I
            # zeros: let the inr be 0 outside I
            # shrink domain: only evaluate points whose ball is contained in I
            # evaluate: sample points outside I
            raise NotImplementedError("TODO: padding modes")

    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.integrator = partial(inrF.conv, inr=new_inr, layer=self)
        new_inr.channels = self.out_channels
        return new_inr
        

class Reshape(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.reshape(-1, *self.dims)
