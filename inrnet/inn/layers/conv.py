import torch, pdb
from functools import partial
nn = torch.nn
F = nn.functional

from inrnet.inn import functional as inrF, polynomials
from scipy.interpolate import RectBivariateSpline as Spline2D

def translate_conv2d(conv2d, img_shape, order=3, smoothing=.01): #h,w
    h,w = img_shape
    out_, in_, k1, k2 = conv2d.weight.shape
    K = k1 / h, k2 / w

    if conv2d.groups != 1:
        raise NotImplementedError("groups")

    if conv2d.stride == (2,2):
        stride = (K[0]*.75, K[1]*.75)
        raise NotImplementedError("stride")
    elif conv2d.stride == (1,1):
        stride = 0.
    else:
        raise NotImplementedError("stride")

    if conv2d.bias is None:
        pass
    else:
        raise NotImplementedError("bias")
        
    layer = inn.SplineConv(kernel_size=K, stride=stride, parameterization="B-spline")
    layer.weight.data = W

    # fit pretrained kernel with b-spline
    x,y = np.linspace(-K[0]/2, K[0]/2, h), np.linspace(-K[1]/2, K[1]/2, w)
    kx = ky = order
    bs = Spline2D(x,y, W.detach().cpu().numpy(), kx=kx,ky=ky, s=smoothing)
    tx,ty,c = [torch.tensor(z).float() for z in bs.tck]
    partial(deBoor2d, args=(tx,ty,c.reshape(h,w),kx,ky))
    layer.interpolator = deBoor2d(qx,qy)
    return layer

def deBoor2d(x,y, args):
    tx,ty, c, px,py = args
    try:
        kx = (tx>x).nonzero()[0].item()-1
    except IndexError:
        kx = tx.size(0)-px-2
    try:
        ky = (ty>y).nonzero()[0].item()-1
    except IndexError:
        ky = ty.size(0)-py-2
        
    d = torch.empty(px+1, py+1)
    for j in range(0, px+1):
        for i in range(0, py+1):
            d[j,i] = c[j+kx-px,i+ky-py]

    for r in range(1, px + 1):
        for j in range(px, r - 1, -1):
            alphax = (x - tx[j + kx - px]) / (tx[j + 1 + kx - r] - tx[j + kx - px])
            for i in range(0, py+1):
                d[j][i] = (1-alphax) * d[j-1][i] + alphax * d[j][i]

    for ry in range(1, py + 1):
        for jy in range(py, ry - 1, -1):
            alphay = (y - ty[jy + ky - py]) / (ty[jy + 1 + ky - ry] - ty[jy + ky - py])
            d[px][jy] = (1-alphay) * d[px][jy-1] + alphay * d[px][jy]
            
    return d[px][py]



class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, input_dims=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_dims = input_dims
        
class SplineConv(Conv):
    def __init__(self, in_channels, out_channels, radius, stride=False, p_norm="inf",
            input_dims=2, N_bins=16, groups=1, bias=False, dropout=0.):
        super().__init__(in_channels, out_channels, input_dims=input_dims)

    def forward(self, inr):
        new_inr = inr.create_derived_inr()
        new_inr.integrator = partial(inrF.spline_conv, inr=new_inr, layer=self)
        new_inr.channels = self.out_channels
        return new_inr

class BallConv(Conv):
    def __init__(self, in_channels, out_channels, radius, stride=False, p_norm="inf",
            input_dims=2, N_bins=16, groups=1, bias=False,
            parameterization="polynomial", padding_mode="cutoff",
            order=3, dropout=0.):
        super().__init__(in_channels, out_channels, input_dims=input_dims)
        self.radius = radius
        self.input_dims = input_dims
        self.dropout = dropout
        self.N_bins = N_bins
        self.stride = stride
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
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

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
