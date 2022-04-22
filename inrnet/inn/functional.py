from scipy.stats.qmc import Sobol
import math, torch, pdb
import numpy as np
nn=torch.nn

### Convolutions

def conv(values, inr, layer, query_coords=None):
    coords = inr.sampled_coords
    if query_coords is None:
        if layer.stride > 0:
            inr.sampled_coords = query_coords = subsample_points_by_grid(coords, spacing=layer.stride)
        else:
            query_coords = coords

    Diffs = query_coords.unsqueeze(0) - coords.unsqueeze(1)
    mask = layer.norm(Diffs) < layer.radius

    if layer.dropout > 0 and inr.training:
        mask *= torch.rand_like(mask, dtype=torch.half) > layer.dropout

    Y = values[torch.where(mask)[1]] # flattened list of values of neighborhood points
    Diffs = Diffs[mask] # flattened tensor of diffs between center coords and neighboring points
    lens = tuple(mask.sum(0)) # number of kernel points assigned to each point
    Ysplit = Y.split(lens) # list of values at neighborhood points
    newVals = []

    if layer.N_bins == 0:
        #Wsplit = layer.K(Diffs).split(lens)
        Dsplit = Diffs.split(lens) # list of diffs of neighborhood points
        for ix,y in enumerate(Ysplit):
            if y.size(0) == 0:
                newVals.append(y.new_zeros(layer.out_channels))
            else:
                W = layer.weight(Dsplit[ix])
                newVals.append(y.unsqueeze(1).matmul(W).squeeze(1).mean(0))

    else:
        bin_centers = get_minNN_points_in_disk(radius=layer.radius, N=layer.N_bins)
        Kh = layer.weight(bin_centers)
        with torch.no_grad():
            bin_ixs = (Diffs.unsqueeze(0) - bin_centers.unsqueeze(1)).norm(dim=-1).min(0).indices # (N_points, d)
        Wsplit = Kh.index_select(dim=0, index=bin_ixs).split(lens)
        
        newVals = []
        for ix,y in enumerate(Ysplit):
            if y.size(0) == 0:
                newVals.append(y.new_zeros(layer.out_channels))
            else:
                newVals.append(y.unsqueeze(1).matmul(Wsplit[ix]).squeeze(1).mean(0))
        # if not inr.training:
        #     for ix,l in enumerate(lens):
        #         if l>1:
        #             print(ix, end=",")
        #         if l==0:
        #             print(ix, end="!")
        #     pdb.set_trace()
        # newVals = [y.unsqueeze(1).matmul(Wsplit[ix]).squeeze(1).mean(0) for ix,y in enumerate(Ysplit)]
    newVals = torch.stack(newVals, dim=0)
    if layer.bias:
        newVals = newVals + layer.bias
    return newVals

def gridconv(values, inr, layer):
    # applies conv once to each patch (token)
    coords = inr.sampled_coords
    n_points = round(2/layer.spacing)
    query_coords = util.meshgrid_coords(n_points, n_points, domain=inr.domain)
    return apply_conv(coords, values, inr, layer, query_coords=query_coords)

def avg_pool(values, inr, layer, query_coords=None):
    coords = inr.sampled_coords
    if query_coords is None:
        if layer.stride > 0:
            query_coords = subsample_points_by_grid(coords, spacing=layer.stride)
            inr.sampled_coords = query_coords
        else:
            query_coords = coords

    Diffs = query_coords.unsqueeze(0) - coords.unsqueeze(1)
    mask = layer.norm(Diffs) < layer.radius 
    Y = values[torch.where(mask)[1]]
    return torch.stack([y.mean(0) for y in Y.split(tuple(mask.sum(0)))])

def adaptive_avg_pool(values, inr, layer):
    coords = inr.sampled_coords
    Diffs = coords.unsqueeze(0) - coords.unsqueeze(1)
    mask = layer.norm(Diffs) < layer.radius 
    Y = values[torch.where(mask)[1]]
    return torch.stack([y.mean(0) for y in Y.split(tuple(mask.sum(0)))])





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

def max_pool(values, inr, layer, query_coords=None):
    coords = inr.sampled_coords
    if query_coords is None:
        if layer.stride > 0:
            query_coords = subsample_points_by_grid(coords, spacing=layer.stride)
            inr.sampled_coords = query_coords
        else:
            query_coords = coords

    Diffs = query_coords.unsqueeze(0) - coords.unsqueeze(1)
    mask = layer.norm(Diffs) < layer.radius
    Y = values[torch.where(mask)[1]]#, inr.input_dims:]
    return torch.stack([y.max(0).values for y in Y.split(tuple(mask.sum(0)))])


def normalize(values, layer, eps=1e-6):
    mean = values.mean(0)
    var = values.pow(2).mean(0) - mean.pow(2)
    if hasattr(layer, "running_mean"):
        if layer.training:
            with torch.no_grad():
                layer.running_mean = layer.momentum * layer.running_mean + (1-layer.momentum) * mean
                layer.running_var = layer.momentum * layer.running_var + (1-layer.momentum) * var
        return (values - layer.running_mean)/(layer.running_var.sqrt() + eps) * layer.learned_std + layer.learned_mean
    elif layer.affine:
        return (values - mean)/(var.sqrt() + eps) * layer.learned_std + layer.learned_mean
    else:
        return (values - mean)/(var.sqrt() + eps)




### Misc

def generate_quasirandom_sequence(d=3, n=128, dtype=torch.float, device="cuda"):
    sobol = Sobol(d=d)
    sample = sobol.random_base2(m=int(math.ceil(math.log2(n))))
    return torch.as_tensor(sample).to(dtype=dtype, device=device)

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

def subsample_points_by_grid(coords, spacing, input_dims=2, domain=(-1,1)):
    bin_ixs = torch.round(coords.squeeze(0)[:,:input_dims] / spacing).int()
    bin_ixs = bin_ixs[:,0]*int(1/spacing*3) + bin_ixs[:,1] # TODO: adapt for larger domains, d
    bins = torch.unique(bin_ixs)
    matches = bin_ixs.unsqueeze(0) == bins.unsqueeze(1) # (bin, point)
    points_per_bin = tuple(matches.sum(1))
    surviving_indices = [x[np.random.randint(0,len(x))] for x in torch.where(matches)[1].split(points_per_bin)]
    return coords[...,surviving_indices,:]

def subsample_points_by_grid(coords, spacing, input_dims=2, domain=(-1,1)):
    bin_ixs = torch.round(coords.squeeze(0)[:,:input_dims] / spacing).int()
    bin_ixs = bin_ixs[:,0]*int(1/spacing*3) + bin_ixs[:,1] # TODO: adapt for larger domains, d
    bins = torch.unique(bin_ixs)
    matches = bin_ixs.unsqueeze(0) == bins.unsqueeze(1) # (bin, point)
    points_per_bin = tuple(matches.sum(1))
    surviving_indices = [x[np.random.randint(0,len(x))] for x in torch.where(matches)[1].split(points_per_bin)]
    return coords[...,surviving_indices,:]


def interpolate(query_coords, observed_coords, values):
    if query_coords.size(0) == 0:
        return values.new_zeros(0, values.size(1))

    dists = (query_coords.unsqueeze(0) - observed_coords.unsqueeze(1)).norm(dim=-1)
    r, indices = dists.topk(3, dim=0, largest=False)
    q = (1/r).unsqueeze(-1)
    Q = q.sum(0)
    sv = (values[indices[0]] * q[0] + values[indices[1]] * q[1] + values[indices[2]] * q[2])/Q
    return sv
