import torch, pdb, math
import numpy as np
from functools import partial
nn = torch.nn
F = nn.functional

from inrnet.inn import functional as inrF, polynomials
from scipy.interpolate import RectBivariateSpline as Spline2D

def translate_conv2d(conv2d, input_shape, extrema, smoothing=.05, **kwargs): #h,w
    # offset/grow so that the conv kernel goes a half pixel past the boundary
    h,w = input_shape # shape of input features/image
    out_, in_, k1, k2 = conv2d.weight.shape
    extrema_dists = extrema[0][1] - extrema[0][0], extrema[1][1] - extrema[1][0]
    spacing = extrema_dists[0] / (h-1), extrema_dists[1] / (w-1)
    K = k1 * spacing[0], k2 * spacing[1]
    order = k1//2+1

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
        stride = 0.
        out_shape = input_shape
    elif conv2d.stride in [2,(2,2)]:
        stride = conv2d.stride[0] * spacing[0], conv2d.stride[1] * spacing[1]
        out_shape = (input_shape[0]//2, input_shape[1]//2)
        extrema = ((extrema[0][0], extrema[0][1]-spacing[0]),
                (extrema[1][0], extrema[1][1]-spacing[1]))
    else:
        raise NotImplementedError("stride")

    bias = conv2d.bias is not None
    layer = SplineConv(in_*conv2d.groups, out_, order=order, smoothing=smoothing,
        init_weights=conv2d.weight.detach().cpu().numpy()*k1*k2,
        # scale up weights since we divide by the number of grid points
        groups=conv2d.groups, padding=conv2d.padding,
        padded_extrema=padded_extrema,
        N_bins=2**math.ceil(math.log2(k1*k2)), shift=shift, control_grid_dims=(k1,k2),
        kernel_size=K, stride=stride, bias=bias, **kwargs)
    if bias:
        layer.bias.data = conv2d.bias.data

    return layer, out_shape, extrema



class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, input_dims=2, stride=0., groups=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_dims = input_dims
        self.stride = stride
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

class SplineConv(Conv):
    def __init__(self, in_channels, out_channels, kernel_size, control_grid_dims, init_weights, order=2, stride=0.,
            input_dims=2, N_bins=0, groups=1, padding=0, padded_extrema=None, bias=False, smoothing=0., shift=(0,0)):
        super().__init__(in_channels, out_channels, input_dims=input_dims,
            stride=stride, bias=bias, groups=groups)
        self.N_bins = N_bins
        self.kernel_size = K = kernel_size
        self.parameterization="B-spline"
        self.dropout = 0
        self.padding = padding
        self.in_groups = self.in_channels // self.groups
        self.register_buffer("padded_extrema", torch.as_tensor(padded_extrema))
        self.register_buffer('shift', torch.tensor(shift, dtype=torch.float))
        if groups == 1 or (groups == in_channels and groups == out_channels):
            pass
        else:
            raise NotImplementedError("groups")

        # fit pretrained kernel with b-spline
        h,w = control_grid_dims
        bbox = (-K[0]/2, K[0]/2, -K[1]/2, K[1]/2)
        x,y = (np.linspace(bbox[0]/h*(h-1), bbox[1]/h*(h-1), h),
               np.linspace(bbox[2]/w*(w-1), bbox[3]/w*(w-1), w))

        self.order = order
        C = []
        for i in range(self.in_groups):
            C.append([])
            for o in range(out_channels):
                bs = Spline2D(x,y, init_weights[o,i], bbox=bbox, kx=order,ky=order, s=smoothing)
                tx,ty,c = [torch.tensor(z).float() for z in bs.tck]
                try:
                    C[-1].append(c.reshape(h,w))
                except RuntimeError:
                    h=tx.size(0)-order-1
                    w=ty.size(0)-order-1
                    C[-1].append(c.reshape(h,w))
            C[-1] = torch.stack(C[-1],dim=0)

        self.register_buffer("grid_points", torch.as_tensor(
            np.dstack(np.meshgrid(x,y)).reshape(-1,2), dtype=torch.float))
        self.register_buffer("sample_points", inrF.generate_quasirandom_sequence(n=N_bins, d=input_dims, bbox=bbox))
        self.register_buffer("Tx", tx)
        self.register_buffer("Ty", ty)
        # self.Tx = nn.Parameter(tx)
        # self.Ty = nn.Parameter(ty)
        self.C = nn.Parameter(torch.stack(C, dim=1))

    def __repr__(self):
        return f"""SplineConv(in_channels={self.in_channels}, out_channels={
        self.out_channels}, kernel_size={np.round(self.kernel_size, decimals=3)}, bias={self.bias is not None})"""

    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.integrator = partial(inrF.conv, inr=new_inr, layer=self)
        new_inr.channels = self.out_channels
        return new_inr

    def kernel_intersection_ratio(self, query_coords):
        dist_to_boundary = (query_coords.unsqueeze(1) - self.padded_extrema.T.unsqueeze(0)).abs().amin(dim=1)
        k = self.kernel_size[0]/2, self.kernel_size[1]/2
        padding_ratio = (self.kernel_size[0] - F.relu(k[0] - dist_to_boundary[:,0])) * (
            self.kernel_size[1] - F.relu(k[1] - dist_to_boundary[:,1])) / self.kernel_size[0] / self.kernel_size[1]
        return padding_ratio

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

        in_, out_ = self.in_groups, self.out_channels
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

        return torch.stack(w_oi).view(xy.size(0), self.out_channels, self.in_groups)

    def cluster_diffs(self, x, tol=1e-5, grid_mode=False):
        """Based on kmeans in https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html"""
        if grid_mode:
            c = self.grid_points  # Initialize centroids to grid
        else:
            c = self.sample_points # Initialize centroids with low-disc seq
        x_i = x.unsqueeze(1)  # (N, 1, D) samples
        c_j = c.unsqueeze(0)  # (1, K, D) centroids

        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        minD, indices = D_ij.min(dim=1)
        cl = indices.view(-1)  # Points -> Nearest cluster
        if grid_mode and minD.mean() > tol:
            print("bad grid alignment")
            pdb.set_trace()

        return cl, c





class BallConv(Conv):
    def __init__(self, in_channels, out_channels, radius, stride=0., p_norm="inf",
            input_dims=2, N_bins=16, groups=1, bias=False,
            parameterization="polynomial", padding_mode="cutoff",
            order=3, dropout=0.):
        super().__init__(in_channels, out_channels, input_dims=input_dims,
            stride=stride, bias=bias, groups=groups)
        self.radius = radius
        self.dropout = dropout
        self.N_bins = N_bins
        if p_norm == "inf":
            p_norm = torch.inf
        self.norm = partial(torch.linalg.norm, ord=p_norm, dim=-1)

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
            # zeros: let the INR be 0 outside I
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
