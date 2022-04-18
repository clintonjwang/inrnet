from scipy.stats.qmc import Sobol
import math, torch, pdb
nn=torch.nn

def conv(coords, values, inr, layer, query_coords=None):
    if query_coords is None:
        if layer.stride > 0:
            query_coords = subsample_points_by_grid(coords, spacing=layer.stride)
            inr.sampled_coords = query_coords
            raise NotImplementedError
        else:
            query_coords = coords

    Diffs = query_coords.unsqueeze(0) - coords.unsqueeze(1)
    mask = layer.norm(Diffs) < layer.radius

    if layer.dropout > 0 and layer.training:
        mask *= torch.rand_like(mask, dtype=torch.half) > layer.dropout

    Y = values[torch.where(mask)[1]] # flattened list of values of neighborhood points
    Diffs = Diffs[mask] # flattened tensor of diffs between center coords and neighboring points
    lens = tuple(mask.sum(0)) # number of kernel points assigned to each point
    Ysplit = Y.split(lens) # list of values at neighborhood points
    newVals = []

    if layer.N_bins == 0:
        #W = layer.K(Diffs) # weights for each pair
        #Wsplit = W.split(lens) # list of weights of neighborhood points
        Dsplit = Diffs.split(lens) # list of diffs of neighborhood points
        for ix,y in enumerate(Ysplit):
            if y.size(0) == 0:
                newVals.append(y.new_zeros(layer.out_channels))
            else:
                W = layer.K(Dsplit[ix])
                newVals.append(y.unsqueeze(1).matmul(W).squeeze(1).mean(0))

    else:
        bin_centers = get_minNN_points_in_disk(radius=layer.radius, N=layer.N_bins)
        Kh = layer.K(bin_centers)
        with torch.no_grad():
            bin_ixs = (Diffs.unsqueeze(0) - bin_centers.unsqueeze(1)).norm(dim=-1).min(0).indices # (N_points, d)
        Wsplit = Kh.index_select(dim=0, index=bin_ixs).split(lens)
        
        newVals = []
        for ix,y in enumerate(Ysplit):
            if y.size(0) == 0:
                newVals.append(y.new_zeros(layer.out_channels))
            else:
                newVals.append(y.unsqueeze(1).matmul(Wsplit[ix]).squeeze(1).mean(0))
        # newVals = [y.unsqueeze(1).matmul(Wsplit[ix]).squeeze(1).mean(0) for ix,y in enumerate(Ysplit)]
    return torch.stack(newVals, dim=0)

def gridconv(coords, values, inr, layer):
    n_points = round(2/layer.stride)
    query_coords = util.meshgrid_coords(n_points, n_points, domain=inr.domain)
    return apply_conv(coords, values, inr, layer, query_coords=query_coords)

def inner_product(coords, values, layer):
    W_ij = layer.m_ij(coords) #(B,d) -> (B,cin,cout)
    if layer.normalized:
        return values.unsqueeze(1).matmul(torch.softmax(W_ij, dim=-1)).squeeze(1) + layer.b_j
    else:
        return values.unsqueeze(1).matmul(W_ij).squeeze(1) + layer.b_j

def avg_pool(coords, values, layer, query_coords=None):
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

def global_avg_pool(values):
    return values.mean(0)

def max_pool(coords, values, layer, query_coords=None):
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


def normalize(coords, values, layer, eps=1e-6):
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




def generate_quasirandom_sequence(d=3, n=128):
    sobol = Sobol(d=d)
    sample = sobol.random_base2(m=int(math.ceil(math.log2(n))))
    return sample

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
    bin_ixs = torch.round(coords / spacing).int()
    bin_ixs = bin_ixs[:,0]*int(1/spacing*3) + bin_ixs[:,1] # TODO: adapt for larger domains, d
    bins = torch.unique(bin_ixs)
    matches = bin_ixs.unsqueeze(0) == bins.unsqueeze(1) # (bin, point)
    points_per_bin = tuple(matches.sum(1))
    surviving_indices = [np.random.choice(x) for x in torch.where(matches)[1].split(points_per_bin)]
    return coords[surviving_indices]
