import numpy as np
import torch, pdb, math
nn = torch.nn
F = nn.functional

zern_radial_coeffs = [
    [(2., 0.)],
    (np.array((2, 0, -1)) * math.sqrt(3), (math.sqrt(6),0,0)),
    (np.array((3, 0, -2, 0)) * math.sqrt(8), (math.sqrt(8),0,0,0)),
    (np.array((6, 0, -6, 0, 1)) * math.sqrt(5),
        np.array((4, 0, -3, 0, 0)) * math.sqrt(10),
        (math.sqrt(10),0,0,0,0)),
    (np.array((10, 0, -12, 0, 3, 0)) * math.sqrt(6),
        np.array((5, 0, -4, 0, 0, 0)) * math.sqrt(12),
        (math.sqrt(12),0,0,0,0,0)),
    (np.array((20, 0, -30, 0, 12, 0, -1)) * math.sqrt(7),
        np.array((15, 0, -20, 0, 6, 0, 0)) * math.sqrt(14),
        np.array((6,0,-5,0,0,0,0)) * math.sqrt(14),
        (math.sqrt(14),0,0,0,0,0,0)),
]
zern_order_indices = ([-1,1], [-2,0,2], [-3,-1,1,3], [-4,-2,0,2,4],
                 [-5,-3,-1,1,3,5], [-6,-4,-2,0,2,4,6])

def get_zern_radial_coeffs(order):
    T = []
    for o in range(1,order+1):
        coeffs = zern_radial_coeffs[o-1]
        if o % 2 == 0:
            C = torch.tensor(np.array([*coeffs[1:][::-1], coeffs[0], *coeffs[1:]]))
        else:
            C = torch.tensor(np.array([*coeffs[::-1], *coeffs]))
        C = torch.cat((C.new_zeros(o+1,order-o), C), dim=1)
        T.append(C)
    return torch.cat(T, dim=0).float()

class ZernikeFunction(nn.Module):
    # basis for L^2 functions on a disk of radius 1 (each basis fxn is bounded by [-1,1])
    # https://en.wikipedia.org/wiki/Zernike_polynomials
    def __init__(self, radius=.2, order=3):
        super().__init__()
        self.order = order
        self.radius = radius
        self.weights = nn.Parameter(torch.randn(sum(range(2,order+2))))
        self.register_buffer('basis_coeffs', get_zern_radial_coeffs(order)) #ocp

        ang_indices = torch.cat([torch.tensor(t) for t in zern_order_indices[:order]]) #op
        self.register_buffer('ix1', ang_indices.abs())
        self.register_buffer('ix2', (ang_indices>=0).int())

    def forward(self, xy):
        r = xy.norm(dim=1) / self.radius
        powers = torch.stack([r.pow(n) for n in range(self.order+1)], dim=-1) #bp
        radial_coeffs = self.weights.unsqueeze(-1) * self.basis_coeffs #op
        radials = powers.unsqueeze(1).matmul(radial_coeffs.T).squeeze(1) #bo

        theta = torch.atan2(xy[:,0],xy[:,1])
        angulars = torch.stack([torch.stack((torch.sin(theta*n), torch.cos(theta*n)), dim=-1) \
                        for n in range(self.order+1)], dim=1) #bp2
        A = torch.stack([angulars[:, p, y] for p, y in zip(self.ix1, self.ix2)], -1) #bo
        return (radials*A).mean(dim=1)


class ZernikeKernel(nn.Module):
    def __init__(self, in_channels, out_channels, radius, order=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.order = order
        self.weights = nn.Parameter(torch.randn(sum(range(2,order+2)), in_channels*out_channels))
        self.register_buffer('basis_coeffs', get_zern_radial_coeffs(order)) #ocp

        ang_indices = torch.cat([torch.tensor(t) for t in zern_order_indices[:order]]) #op
        self.register_buffer('ix1', ang_indices.abs())
        self.register_buffer('ix2', (ang_indices<0).int())

    def forward(self, xy):
        r = xy.norm(dim=1) / self.radius
        powers = torch.stack([r.pow(n) for n in range(self.order+1)], dim=-1) #bp
        radial_coeffs = self.weights.unsqueeze(1) * self.basis_coeffs.unsqueeze(-1)
        radials = torch.einsum("bp,opc->boc", powers, radial_coeffs) #boc

        theta = torch.atan2(xy[:,0],xy[:,1])
        angulars = torch.stack([torch.stack((torch.sin(theta*n), torch.cos(theta*n)), dim=-1) \
                        for n in range(self.order+1)], dim=1) #bp2
        A = torch.stack([angulars[:, p, y] for p, y in zip(self.ix1, self.ix2)], -1) #bo

        return (radials*A.unsqueeze(-1)).mean(dim=1).reshape(-1, self.in_channels, self.out_channels)


class CircularHarmonics(nn.Module):
    def __init__(self, in_channels, out_channels, radius, order):
        super().__init__()
        # a map from (x,y) -> Wij
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.order = order
        self.weight = nn.Parameter(torch.ones(1,in_channels,out_channels))
        # apply the same filter to each input channel
        self.W = nn.Parameter(torch.randn(1,1, out_channels, order, 2) / torch.arange(1,order+1).reshape(1,1,1,order,1))
        self.var = nn.Parameter(torch.full([1,1,out_channels], radius**2))

    def forward(self, xy):
        xy = xy.view(-1,2,1,1,1)
        x,y = xy[:,0], xy[:,1]
        R = (-.5*(x*x+y*y)/self.var).exp()
        theta = torch.atan2(x,y)
        A = sum([self.W[...,n,0] * torch.cos(n*theta) + self.W[...,n,1] * torch.sin(n*theta) for n in range(self.order)])
        return (self.weight*R*A).reshape(-1, self.in_channels, self.out_channels)



leg_basis_coeffs = [
    (1, 0),
    (1.5, 0, -.5),
    (2.5, 0, -1.5, 0),
    np.array((35, 0, -30, 0, 3))/8,
    np.array((63, 0, -70, 0, 15, 0))/8,
    np.array((231, 0, -315, 0, 105, 0, -5))/16,
]

class LegendreFunction(nn.Module):
    # a function from [-1,1]^d->R, expressed as the product of
    # d 1D polynomials, each the sum of n-order Legendre polynomials
    # the Legendre polynomials are the uniquely orthogonal basis on [-1,1] (wrt constant weights)
    def __init__(self, input_dims=2, order=6):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1).float())
        self.weights = nn.Parameter(torch.randn(input_dims, order))
        basis_coeffs = []
        for o in range(order):
            basis_coeffs.append([0]*(order-o-1) + list(leg_basis_coeffs[o]))
        self.register_buffer('basis_coeffs', torch.tensor(basis_coeffs).float())
        self.order = order
        if order > 6:
            raise NotImplementedError

    def forward(self, x):
        powers = torch.stack([x.pow(n) for n in range(self.order+1)], dim=-1) #bxp
        weighted_coeffs = self.weights.matmul(self.basis_coeffs) #xw,wp->xp
        return torch.einsum("bxp,xp->b", powers, weighted_coeffs)/self.order + self.bias


class LegendreFilter(nn.Module):
    # m LegendreFunctions [-r,r]^d->R^m
    def __init__(self, in_channels, out_channels, radius=1., order=6, input_dims=2):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(in_channels*out_channels, dtype=torch.float))
        self.weights = nn.Parameter(torch.randn(in_channels*out_channels,
            input_dims, order) / input_dims / math.sqrt(in_channels+out_channels))
        self.order = order
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        basis_coeffs = []
        for o in range(order):
            basis_coeffs.append([0]*(order-o-1) + list(leg_basis_coeffs[o]))
        self.register_buffer('basis_coeffs', torch.tensor(basis_coeffs, dtype=torch.float))
        if order > 6:
            raise NotImplementedError

    def forward(self, x):
        powers = torch.stack([(x/self.radius).pow(n) for n in range(self.order+1)], dim=-1) #bxp
        weighted_coeffs = self.weights.matmul(self.basis_coeffs) #cxw,wp->cxp
        outputs = torch.einsum("bxp,cxp->bc", powers, weighted_coeffs)/self.order + self.bias
        return outputs.reshape(-1, self.in_channels, self.out_channels)

