from scipy.stats.qmc import Sobol
import math, torch, pdb
import numpy as np
nn=torch.nn

### Convolutions

def conv(values, inr, layer, query_coords=None):
    coords = inr.sampled_coords
    if query_coords is None:
        if layer.stride != 0:
            inr.sampled_coords = query_coords = subsample_points_by_grid(coords, spacing=layer.stride)
        else:
            query_coords = coords

    if hasattr(layer, "norm"):
        Diffs = query_coords.unsqueeze(1) - coords.unsqueeze(0)
        mask = layer.norm(Diffs) < layer.radius
    else:
        if hasattr(layer, "shift"):
            query_coords = query_coords + torch.tensor(layer.shift, dtype=query_coords.dtype, device=query_coords.device)
        Diffs = query_coords.unsqueeze(1) - coords.unsqueeze(0)
        mask = (Diffs[...,0].abs() < layer.kernel_size[0]/2) * (Diffs[...,1].abs() < layer.kernel_size[1]/2)
        
    # Diffs = query_coords.unsqueeze(0) - coords.unsqueeze(1)
    # mask = layer.norm(Diffs) < layer.radius
    if layer.dropout > 0 and (inr.training and layer.training):
        mask *= torch.rand_like(mask, dtype=torch.half) > layer.dropout

    Y = values[torch.where(mask)[1]] # flattened list of values of neighborhood points
    Diffs = Diffs[mask] # flattened tensor of diffs between center coords and neighboring points
    lens = tuple(mask.sum(1)) # number of kernel points assigned to each point
    Ysplit = Y.split(lens) # list of values at neighborhood points
    newVals = []

    if hasattr(layer, "interpolate_weights"):
        if layer.N_bins == 0:
            Dsplit = Diffs.split(lens) # list of diffs of neighborhood points
            if layer.groups != 1:
                for ix,y in enumerate(Ysplit):
                    w_o = layer.interpolate_weights(-Dsplit[ix]).squeeze(-1)
                    newVals.append(torch.einsum('ni,ni->i',y,w_o))
            else:
                for ix,y in enumerate(Ysplit):
                    w_oi = layer.interpolate_weights(-Dsplit[ix])
                    newVals.append(torch.einsum('ni,noi->o',y,w_oi))

        else:
            bin_ixs, bin_centers = layer.kmeans(Diffs)
            if layer.groups != 1:
                w_o = layer.interpolate_weights(-bin_centers).squeeze(-1)
                Wsplit = w_o.index_select(dim=0, index=bin_ixs).split(lens)
                for ix,y in enumerate(Ysplit):
                    newVals.append(torch.einsum('bi,bi->i',y,Wsplit[ix]))
            else:
                w_oi = layer.interpolate_weights(-bin_centers)
                Wsplit = w_oi.index_select(dim=0, index=bin_ixs).split(lens)
                for ix,y in enumerate(Ysplit):
                    newVals.append(torch.einsum('bi,boi->o',y,Wsplit[ix]))

    else:
        if layer.N_bins == 0:
            #Wsplit = layer.K(Diffs).split(lens)
            Dsplit = Diffs.split(lens) # list of diffs of neighborhood points
            for ix,y in enumerate(Ysplit):
                if y.size(0) == 0:
                    newVals.append(y.new_zeros(layer.out_channels))
                else:
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
                if y.size(0) == 0:
                    newVals.append(y.new_zeros(layer.out_channels))
                else:
                    newVals.append(y.unsqueeze(1).matmul(Wsplit[ix]).squeeze(1).mean(0))
                    
    newVals = torch.stack(newVals, dim=0)

    if layer.bias is not None:
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
        if layer.stride != 0:
            inr.sampled_coords = query_coords = subsample_points_by_grid(coords, spacing=layer.stride)
        else:
            query_coords = coords

    Diffs = query_coords.unsqueeze(1) - coords.unsqueeze(0)
    mask = layer.norm(Diffs) < layer.radius 
    Y = values[torch.where(mask)[1]]
    return torch.stack([y.mean(0) for y in Y.split(tuple(mask.sum(1)))])


def max_pool(values, inr, layer, query_coords=None):
    coords = inr.sampled_coords
    if query_coords is None:
        if layer.stride != 0:
            inr.sampled_coords = query_coords = subsample_points_by_grid(coords, spacing=layer.stride)
        else:
            query_coords = coords

    if hasattr(layer, "norm"):
        Diffs = query_coords.unsqueeze(1) - coords.unsqueeze(0)
        mask = layer.norm(Diffs) < layer.radius
    else:
        query_coords = query_coords + torch.tensor(layer.shift, dtype=query_coords.dtype, device=query_coords.device)
        Diffs = query_coords.unsqueeze(1) - coords.unsqueeze(0)
        mask = (Diffs[...,0].abs() < layer.kernel_size[0]/2) * (Diffs[...,1].abs() < layer.kernel_size[1]/2)
    lens = tuple(mask.sum(1))
    Y = values[torch.where(mask)[1]]
    Ysplit = Y.split(lens)
    return torch.stack([y.max(0).values for y in Ysplit], dim=0)

#((Diffs[:,-1,0].abs() <= layer.kernel_size[0]) * (Diffs[:,-1,1].abs() <= layer.kernel_size[1])).sum()



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


def normalize(values, inr, layer):
    mean = values.mean(0)
    var = values.pow(2).mean(0) - mean.pow(2)
    if hasattr(layer, "running_mean"):
        if inr.training and layer.training:
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
        def select_topleft(indices):
            return indices[torch.min(coords[indices,0] + coords[indices,1]*.1, dim=0).indices.item()]
        surviving_indices = [select_topleft(x) for x in torch.where(matches)[1].split(points_per_bin)]

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
