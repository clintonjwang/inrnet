from scipy.stats.qmc import Sobol
import math, torch, pdb
nn=torch.nn

def generate_quasirandom_sequence(d=3, n=128):
    sobol = Sobol(d=d)
    sample = sobol.random_base2(m=int(math.ceil(math.log2(n))))
    return sample

def get_ball_volume(r, dims):
    return ((2.*math.pi**(dims/2.))/(dims*math.gamma(dims/2.)))*r**dims

def apply_conv(inr, layer):
    X = inr.sampled_coords[:,:inr.input_dims]
    Diffs = X.unsqueeze(0) - X.unsqueeze(1)
    mask = layer.norm(Diffs) < layer.radius
    if layer.dropout > 0 and layer.training:
        mask *= torch.rand_like(mask) > layer.dropout
        return
    Y = inr.sampled_coords[torch.where(mask)[1], 2:] # flattened list of values of neighborhood points
    Diffs = Diffs[mask] # flattened tensor of diffs between center coords and neighboring points
    W = layer.K(Diffs).reshape(-1, layer.in_channels, layer.out_channels) # weights for each pair
    lens = tuple(mask.sum(0)) # number of kernel points assigned to each point
    Wsplit = W.split(lens) # list of diffs of neighborhood points
    Ysplit = Y.split(lens) # list of values at neighborhood points
    return torch.stack([y.unsqueeze(1).matmul(Wsplit[ix]).squeeze(1).mean(0) for ix,y in enumerate(Ysplit)], dim=0)

def avg_pool(inr, layer):
    X = inr.sampled_coords[:,:inr.input_dims]
    Diffs = X.unsqueeze(0) - X.unsqueeze(1)
    mask = layer.norm(Diffs) < layer.radius 
    Y = inr.sampled_coords[torch.where(mask)[1], inr.input_dims:]
    return torch.stack([y.mean(0) for y in Y.split(tuple(mask.sum(0)))])

def max_pool(inr, layer):
    X = inr.sampled_coords[:,:inr.input_dims]
    Diffs = X.unsqueeze(0) - X.unsqueeze(1)
    mask = layer.norm(Diffs) < layer.radius
    Y = inr.sampled_coords[torch.where(mask)[1], inr.input_dims:]
    return torch.stack([y.max(0) for y in Y.split(tuple(mask.sum(0)))])

def normalize(inr, layer, eps=1e-6):
    V = inr.sampled_coords[:,inr.input_dims:]
    mean = inr.mean()
    var = inr.pow(2).mean() - mean.pow(2)
    if hasattr(layer, "running_mean"):
        if layer.training:
            with torch.no_grad():
                layer.running_mean = layer.momentum * layer.running_mean + (1-layer.momentum) * mean
                layer.running_var = layer.momentum * layer.running_var + (1-layer.momentum) * var
        return (V - layer.running_mean)/(layer.running_var.sqrt() + eps) * layer.learned_std + layer.learned_mean
    elif layer.affine:
        return (V - mean)/(var.sqrt() + eps) * layer.learned_std + layer.learned_mean
    else:
        return (V - mean)/(var.sqrt() + eps)
