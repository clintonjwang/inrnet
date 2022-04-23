import torch, pdb
import numpy as np
from functools import partial
nn = torch.nn
F = nn.functional

from inrnet.inn import functional as inrF, polynomials
from scipy.interpolate import RectBivariateSpline as Spline2D

def translate_conv2d(conv2d, img_shape, order=2, smoothing=.01, **kwargs): #h,w
    # offset/grow so that the conv kernel goes a half pixel past the boundary
    h,w = img_shape
    out_, in_, k1, k2 = conv2d.weight.shape
    domain_width = 2
    spacing = domain_width / (h-1), domain_width / (w-1)
    K = k1 * spacing[0], k2 * spacing[1]

    if k1 == k2 == 1:
        raise NotImplementedError("ChannelMixer")

    if k1 % 2 == k2 % 2 == 0:
        shift = k1/4, k2/4
        raise NotImplementedError("shift bbox")
    else:
        shift = 0,0

    if conv2d.groups != 1:
        raise NotImplementedError("groups")

    if conv2d.padding != (k1//2,k2//2):
        raise NotImplementedError("padding")

    if conv2d.stride == (1,1):
        stride = 0.
    else:
        stride = conv2d.stride[0] / h * 2.01, conv2d.stride[1] / w * 2.01
        # stride = (K[0]*.75, K[1]*.75)
        raise NotImplementedError("stride")

    if conv2d.bias is None:
        bias = False
    else:
        bias = True
        
    layer = SplineConv(in_, out_, order=order, smoothing=smoothing,
        init_weights=conv2d.weight.detach().cpu().numpy(),
        control_grid_dims=(k1,k2), kernel_size=K, stride=stride, bias=bias, **kwargs)
    if bias:
        layer.bias.data = conv2d.bias

    return layer



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
            input_dims=2, N_bins=16, groups=1, bias=False, smoothing=0., shift=0, padding_mode="cutoff"):
        super().__init__(in_channels, out_channels, input_dims=input_dims,
            stride=stride, bias=bias, groups=groups)
        self.N_bins = N_bins
        self.kernel_size = K = kernel_size
        self.parameterization="B-spline"
        self.dropout = 0
        self.padding_mode = padding_mode

        # fit pretrained kernel with b-spline
        h,w = control_grid_dims
        bbox = (-K[0]/2, K[0]/2, -K[1]/2, K[1]/2)
        if (h == w == 3):
            x,y = np.linspace(bbox[0]/1.5, bbox[1]/1.5, h), np.linspace(bbox[2]/1.5, bbox[3]/1.5, w)
        else:
            raise NotImplementedError("bbox")
        self.order = order
        Tx,Ty,C = [],[],[]
        for i in range(in_channels):
            Tx.append([])
            Ty.append([])
            C.append([])
            for o in range(out_channels):
                bs = Spline2D(x,y, init_weights[o,i], bbox=bbox, kx=order,ky=order, s=smoothing)
                tx,ty,c = [torch.tensor(z).float() for z in bs.tck]
                Tx[-1].append(tx)
                Ty[-1].append(ty)
                C[-1].append(c.reshape(h,w))
            Tx[-1] = torch.stack(Tx[-1],dim=0)
            Ty[-1] = torch.stack(Ty[-1],dim=0)
            C[-1] = torch.stack(C[-1],dim=0)

        self.Tx = nn.Parameter(torch.stack(Tx, dim=1))
        self.Ty = nn.Parameter(torch.stack(Ty, dim=1))
        self.C = nn.Parameter(torch.stack(C, dim=1))

    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.integrator = partial(inrF.conv, inr=new_inr, layer=self)
        new_inr.channels = self.out_channels
        return new_inr

    def interpolate_weights(self, xy):
        w_oi = []
        X = xy[:,0].unsqueeze(1)
        Y = xy[:,1].unsqueeze(1)
        px = py = self.order

        values, kx = (self.Tx<=X.reshape(-1,1,1,1)).min(dim=-1)
        kx -= 1
        kx[values] = self.Tx.size(-1)-px-2

        values, ky = (self.Ty<=Y.reshape(-1,1,1,1)).min(dim=-1)
        ky -= 1
        ky[values] = self.Ty.size(-1)-py-2

        in_, out_ = self.in_channels, self.out_channels
        Dim = in_*out_
        Ctrl = self.C.view(Dim, *self.C.shape[-2:])
        kx = kx.view(-1, Dim)
        ky = ky.view(-1, Dim)
        Tx = self.Tx.view(Dim, -1)
        Ty = self.Ty.view(Dim, -1)
        for z in range(X.size(0)):
            for i in range(Dim):
                D = Ctrl[i, kx[z,i]-px : kx[z,i]+1, ky[z,i]-py : ky[z,i]+1].clone()

                for r in range(1, px + 1):
                    alphax = (X[z,0] - Tx[i,kx[z,i]-px+1:kx[z,i]+1]) / (
                        Tx[i,2+kx[z,i]-r:2+kx[z,i]-r+px] - Tx[i,kx[z,i]-px+1:kx[z,i]+1])
                    for j in range(px, r - 1, -1):
                        D[j] = (1-alphax[j-1]) * D[j-1] + alphax[j-1] * D[j]

                for r in range(1, py + 1):
                    alphay = (Y[z,0] - Ty[i,ky[z,i]-py+1:ky[z,i]+1]) / (
                        Ty[i,2+ky[z,i]-r:2+ky[z,i]-r+py] - Ty[i,ky[z,i]-py+1:ky[z,i]+1])
                    for j in range(py, r-1, -1):
                        D[px][j] = (1-alphay[j-1]) * D[px][j-1] + alphay[j-1] * D[px][j]
                
                w_oi.append(D[px][py])

        # for i in range(self.in_channels):
        #     for o in range(self.out_channels):
        #         D = []
        #         for z in range(X.size(0)):
        #             d = self.C[o,i,kx[z,o,i]-px:kx[z,o,i]+1,ky[z,o,i]-py:ky[z,o,i]+1].clone()

        #             for r in range(1, px + 1):
        #                 alphax = (X[z,0] - self.Tx[o,i,kx[z,o,i]-px+1:kx[z,o,i]+1]) / (
        #                     self.Tx[o,i,2+kx[z,o,i]-r:2+kx[z,o,i]-r+px] - self.Tx[o,i,kx[z,o,i]-px+1:kx[z,o,i]+1])
        #                 for j in range(px, r - 1, -1):
        #                     d[j] = (1-alphax[j-1]) * d[j-1] + alphax[j-1] * d[j]

        #             for r in range(1, py + 1):
        #                 alphay = (Y[z,0] - self.Ty[o,i,ky[z,o,i]-py+1:ky[z,o,i]+1]) / (
        #                     self.Ty[o,i,2+ky[z,o,i]-r:2+ky[z,o,i]-r+py] - self.Ty[o,i,ky[z,o,i]-py+1:ky[z,o,i]+1])
        #                 for j in range(py, r-1, -1):
        #                     d[px][j] = (1-alphay[j-1]) * d[px][j-1] + alphay[j-1] * d[px][j]
                    
        #             D.append(d[px][py])

        #         w_io.append(torch.stack(D))

        return torch.stack(w_oi).reshape(xy.size(0), self.out_channels, self.in_channels)

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
