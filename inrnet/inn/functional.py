from scipy.stats import qmc
import math, torch, pdb
import numpy as np
nn=torch.nn

### Convolutions
class Integrator:
    def __init__(self, function, name, inr=None, layer=None, **kwargs):
        self.function = function
        self.name = name
        self.inr = inr
        self.layer = layer
        self.kwargs = kwargs
    def __repr__(self):
        return self.name
    def __call__(self, values, *args):
        kwargs = self.kwargs.copy()
        if self.inr is not None:
            kwargs['inr'] = self.inr
        if self.layer is not None:
            kwargs['layer'] = self.layer
        return self.function(values, *args, **kwargs)


def conv(values: torch.Tensor, # [B,N,c_in]
    inr, layer: nn.Module,
    query_coords=None):

    dtype = layer.dtype
    coords = inr.sampled_coords.to(dtype=dtype) #[N,d]
    if query_coords is None:
        if layer.stride != 0:
            if inr.grid_mode:
                inr.sampled_coords = query_coords = subsample_points_by_grid(coords, spacing=layer.stride).to(dtype=dtype)
            else:
                inr.sampled_coords = query_coords = coords[:coords.size(0)//4]
        else:
            query_coords = coords

    if hasattr(layer, "norm"):
        Diffs = query_coords.unsqueeze(1) - coords.unsqueeze(0)
        mask = layer.norm(Diffs) < layer.radius
    else:
        center_coords = query_coords + layer.shift
        Diffs = center_coords.unsqueeze(1) - coords.unsqueeze(0)
        mask = (Diffs[...,0].abs() < layer.kernel_size[0]/2) * (Diffs[...,1].abs() < layer.kernel_size[1]/2)
        padding_ratio = layer.kernel_intersection_ratio(center_coords)
        # scaling factor

    # if layer.dropout > 0 and (inr.training and layer.training):
    #     mask *= torch.rand_like(mask, dtype=torch.half) > layer.dropout

    Y = values[:,torch.where(mask)[1]].to(dtype=dtype) # flattened list of values of neighborhood points
    Diffs = Diffs[mask] # flattened tensor of diffs between center coords and neighboring points
    lens = tuple(mask.sum(1)) # number of kernel points assigned to each point
    Ysplit = Y.split(lens, dim=1) # list of values at neighborhood points
    newVals = []

    if hasattr(layer, "interpolate_weights"):
        if inr.grid_mode or layer.N_bins != 0:
            bin_ixs, bin_centers = cluster_diffs(Diffs, layer=layer, grid_mode=inr.grid_mode)

            if layer.groups != 1:
                w_o = layer.interpolate_weights(-bin_centers).squeeze(-1)
                Wsplit = w_o.index_select(dim=0, index=bin_ixs).split(lens)
                for ix,y in enumerate(Ysplit):
                    newVals.append(torch.einsum('bni,ni->bi',y,Wsplit[ix])/y.size(1))
            else:
                w_oi = layer.interpolate_weights(-bin_centers)
                Wsplit = w_oi.index_select(dim=0, index=bin_ixs).split(lens)
                for ix,y in enumerate(Ysplit):
                    newVals.append(torch.einsum('bni,noi->bo',y,Wsplit[ix])/y.size(1))
                    
        else:
            Dsplit = Diffs.split(lens) # list of diffs of neighborhood points
            if layer.groups != 1:
                for ix,y in enumerate(Ysplit):
                    w_o = layer.interpolate_weights(-Dsplit[ix]).squeeze(-1)
                    newVals.append(torch.einsum('bni,ni->bi',y,w_o)/y.size(1))
            else:
                for ix,y in enumerate(Ysplit):
                    w_oi = layer.interpolate_weights(-Dsplit[ix])
                    newVals.append(torch.einsum('bni,noi->bo',y,w_oi)/y.size(1))
        
    else:
        if layer.N_bins == 0:
            #Wsplit = layer.K(Diffs).split(lens)
            Dsplit = Diffs.split(lens) # list of diffs of neighborhood points
            for ix,y in enumerate(Ysplit):
                if y.size(1) == 0:
                    newVals.append(y.new_zeros(y.size(0), layer.out_channels))
                else:
                    raise NotImplementedError('convball')
                    W = layer.weight(-Dsplit[ix])
                    newVals.append(y.unsqueeze(1).matmul(W).squeeze(1).mean(0))

        else:
            bin_centers = get_minNN_points_in_disk(radius=layer.radius, N=layer.N_bins)
            Kh = layer.weight(bin_centers)
            with torch.no_grad():
                bin_ixs = (Diffs.unsqueeze(0) - bin_centers.unsqueeze(1)).norm(dim=-1).min(0).indices # (N_points, d)
            Wsplit = Kh.index_select(dim=0, index=bin_ixs).split(lens)
            
            newVals = []
            for ix,y in enumerate(Ysplit):
                raise NotImplementedError('convball')
                if y.size(1) == 0:
                    newVals.append(y.new_zeros(layer.out_channels))
                else:
                    newVals.append(y.unsqueeze(1).matmul(Wsplit[ix]).squeeze(1).mean(0))
                    
    newVals = torch.stack(newVals, dim=1) #[B,N,c_out]
    newVals *= padding_ratio.unsqueeze(-1)

    if layer.bias is not None:
        newVals = newVals + layer.bias
    return newVals


def cluster_diffs(x, layer, tol=.005, grid_mode=False):
    """Based on kmeans in https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html"""
    if grid_mode:
        c = layer.grid_points  # Initialize centroids to grid
    else:
        c = layer.sample_points # Initialize centroids with low-disc seq
    x_i = x.unsqueeze(1)  # (N, 1, D) samples
    c_j = c.unsqueeze(0)  # (1, K, D) centroids

    D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) squared distances
    minD, indices = D_ij.min(dim=1)
    cl = indices.view(-1)  # Points -> Nearest cluster
    if grid_mode and minD.mean() > tol:
        print("bad grid alignment")
        pdb.set_trace()

    return cl, c

def interpolate_weights_single_channel(xy, tx,ty,c, order=2):
    W = []
    X = xy[:,0].unsqueeze(1)
    Y = xy[:,1].unsqueeze(1)
    px = py = order

    values, kx = (tx<=X).min(dim=-1)
    values, ky = (ty<=Y).min(dim=-1)
    kx -= 1
    ky -= 1
    kx[values] = tx.size(-1)-px-2
    ky[values] = ty.size(-1)-py-2

    for z in range(X.size(0)):
        D = c[kx[z]-px : kx[z]+1, ky[z]-py : ky[z]+1].clone()

        for r in range(1, px + 1):
            try:
                alphax = (X[z,0] - tx[kx[z]-px+1:kx[z]+1]) / (
                    tx[2+kx[z]-r:2+kx[z]-r+px] - tx[kx[z]-px+1:kx[z]+1])
            except RuntimeError:
                print("input off the grid")
                pdb.set_trace()
            for j in range(px, r - 1, -1):
                D[j] = (1-alphax[j-1]) * D[j-1] + alphax[j-1] * D[j].clone()

        for r in range(1, py + 1):
            alphay = (Y[z,0] - ty[ky[z]-py+1:ky[z]+1]) / (
                ty[2+ky[z]-r:2+ky[z]-r+py] - ty[ky[z]-py+1:ky[z]+1])
            for j in range(py, r-1, -1):
                D[px,j] = (1-alphay[j-1]) * D[px,j-1].clone() + alphay[j-1] * D[px,j].clone()
        
        W.append(D[px,py])
    return torch.stack(w_oi)

# def gridconv(values, inr, layer):
#     # applies conv once to each patch (token)
#     coords = inr.sampled_coords
#     n_points = round(2/layer.spacing)
#     query_coords = util.meshgrid_coords(n_points, n_points, domain=inr.domain)
#     return apply_conv(coords, values, inr, layer, query_coords=query_coords)

def avg_pool(values, inr, layer, query_coords=None):
    coords = inr.sampled_coords
    if query_coords is None:
        if layer.stride != 0:
            if inr.grid_mode:
                inr.sampled_coords = query_coords = subsample_points_by_grid(coords, spacing=layer.stride)
            else:
                inr.sampled_coords = query_coords = coords[:coords.size(0)//4]
        else:
            query_coords = coords

    Diffs = query_coords.unsqueeze(1) - coords.unsqueeze(0)
    mask = layer.norm(Diffs) < layer.radius 
    Y = values[:,torch.where(mask)[1]]
    return torch.stack([y.mean(1) for y in Y.split(tuple(mask.sum(1)), dim=1)])


def max_pool(values, inr, layer, query_coords=None):
    coords = inr.sampled_coords
    if query_coords is None:
        if layer.stride != 0:
            if inr.grid_mode:
                inr.sampled_coords = query_coords = subsample_points_by_grid(coords, spacing=layer.stride)
            else:
                inr.sampled_coords = query_coords = coords[:coords.size(0)//4]
        else:
            query_coords = coords

    if torch.amax(layer.shift) > 0:
        query_coords = query_coords + layer.shift
    Diffs = query_coords.unsqueeze(1) - coords.unsqueeze(0)
    mask = Diffs.norm(dim=-1).topk(k=layer.k, dim=1, largest=False).indices
    return values[:,mask].max(dim=2).values



### Integrations over I

def inner_product(values, inr, layer):
    W_ij = layer.m_ij(inr.sampled_coords) #(B,d) -> (B,cin,cout)
    if layer.normalized:
        out = values.unsqueeze(1).matmul(torch.softmax(W_ij, dim=-1)).squeeze(1)
    else:
        out = values.unsqueeze(1).matmul(W_ij).squeeze(1)
    if hasattr(layer, "b_j"):
        return out + layer.b_j
    else:
        return out

def global_avg_pool(values):
    return values.mean(0, keepdim=True)

def adaptive_avg_pool(values, inr, layer):
    coords = inr.sampled_coords
    Diffs = coords.unsqueeze(0) - coords.unsqueeze(1)
    mask = layer.norm(Diffs) < layer.radius 
    Y = values[torch.where(mask)[1]]
    return torch.stack([y.mean(0) for y in Y.split(tuple(mask.sum(0)))])


def inst_normalize(values, inr, layer):
    if hasattr(layer, "running_mean") and not (inr.training and layer.training):
        mean = layer.running_mean
        var = layer.running_var
    else:
        mean = values.mean(1, keepdim=True)
        var = values.pow(2).mean(1, keepdim=True) - mean.pow(2)
        if hasattr(layer, "running_mean"):
            with torch.no_grad():
                layer.running_mean = layer.momentum * layer.running_mean + (1-layer.momentum) * mean.mean()
                layer.running_var = layer.momentum * layer.running_var + (1-layer.momentum) * var.mean()
            mean = layer.running_mean
            var = layer.running_var
    if hasattr(layer, "weight"):
        return (values - mean)/(var.sqrt() + layer.eps) * layer.weight + layer.bias
    else:
        return (values - mean)/(var.sqrt() + layer.eps)


def batch_normalize(values, inr, layer):
    if hasattr(layer, "running_mean") and not (inr.training and layer.training):
        mean = layer.running_mean
        var = layer.running_var
    else:
        mean = values.mean(dim=(0,1))
        var = values.pow(2).mean(dim=(0,1)) - mean.pow(2)
        if hasattr(layer, "running_mean"):
            with torch.no_grad():
                layer.running_mean = layer.momentum * layer.running_mean + (1-layer.momentum) * mean
                layer.running_var = layer.momentum * layer.running_var + (1-layer.momentum) * var
            mean = layer.running_mean
            var = layer.running_var

    if hasattr(layer, "weight"):
        return (values - mean)/(var.sqrt() + layer.eps) * layer.weight + layer.bias
    else:
        return (values - mean)/(var.sqrt() + layer.eps)





### Misc


def generate_quasirandom_sequence(d=2, n=128, bbox=(-1,1,-1,1), scramble=False,
        like=None, dtype=torch.float, device="cuda"):
    if math.log2(n) % 1 == 0:
        sampler = qmc.Sobol(d=d, scramble=scramble)
        sample = sampler.random_base2(m=int(math.log2(n)))
    else:
        sampler = qmc.Halton(d=d, scramble=scramble)
        sample = sampler.random(n=n)
    if like is None:
        out = torch.as_tensor(sample, dtype=dtype, device=device)
    else:
        out = torch.as_tensor(sample, dtype=like.dtype, device=like.device)
    if bbox is not None:
        # bbox has form (x1,x2, y1,y2) in 2D
        out[:,0] = out[:,0] * (bbox[1]-bbox[0]) + bbox[0]
        out[:,1] = out[:,1] * (bbox[3]-bbox[2]) + bbox[2]
    return out

def get_minNN_points_in_disk(N, radius=1., eps=0., dtype=torch.float, device="cuda"):
    # what we really want is a Voronoi partition that minimizes the
    # difference between the smallest and largest cell volumes, and includes (0,0)
    #
    # Fibonacci lattice
    # http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/#more-3069
    indices = torch.arange(0, N).to(device=device, dtype=dtype) + eps
    R = radius*(indices/(N-1+2*eps)).sqrt() * torch.sigmoid(torch.tensor(N).pow(.4))
    # shrink radius by some amount to increase Voronoi cells of outer points
    theta = torch.pi * (1 + 5**0.5) * indices
    return torch.stack((R*torch.cos(theta), R*torch.sin(theta)), dim=1)

def get_ball_volume(r, dims):
    return ((2.*math.pi**(dims/2.))/(dims*math.gamma(dims/2.)))*r**dims

def subsample_points_by_grid(coords, spacing, input_dims=2, random=False):
    x = coords[...,0] / spacing[0]
    y = coords[...,1] / spacing[1]
    x -= x.min()
    y -= y.min()
    bin_ixs = torch.floor(torch.stack((x,y), dim=-1)+1e-4).int()
    bin_ixs = bin_ixs[:,0]*int(3/spacing[1]) + bin_ixs[:,1] # TODO: adapt for larger domains, d
    bins = torch.unique(bin_ixs)
    matches = bin_ixs.unsqueeze(0) == bins.unsqueeze(1) # (bin, point)
    points_per_bin = tuple(matches.sum(1))
    if random:
        surviving_indices = [x[np.random.randint(0,len(x))] for x in torch.where(matches)[1].split(points_per_bin)]
    else:
        surviving_indices = [x[0] for x in torch.where(matches)[1].split(points_per_bin)]
        # def select_topleft(indices):
        #     return indices[torch.min(coords[indices,0] + coords[indices,1]*.1, dim=0).indices.item()]
        # surviving_indices = [select_topleft(x) for x in torch.where(matches)[1].split(points_per_bin)]

    return coords[surviving_indices,:]


def interpolate(query_coords, observed_coords, values):
    if query_coords.size(0) == 0:
        return values.new_zeros(0, values.size(1))

    dists = (query_coords.unsqueeze(0) - observed_coords.unsqueeze(1)).norm(dim=-1)
    r, indices = dists.topk(3, dim=0, largest=False)
    q = (1/r).unsqueeze(-1)
    Q = q.sum(0)
    sv = (values[indices[0]] * q[0] + values[indices[1]] * q[1] + values[indices[2]] * q[2])/Q
    return sv
