"""Functions"""
import pdb
from typing import Callable, Optional
import torch

from inrnet.inn.inr import INRBatch
from inrnet.inn.layers.other import PositionalEncoding
from inrnet.inn.layers.pooling import AvgPool, MaxPool
from inrnet.inn.point_set import PointSet
nn=torch.nn

from inrnet.inn import point_set

def tokenization(values: torch.Tensor, inr: INRBatch):
    """tokenization"""
    return
    
def pos_enc(values: torch.Tensor, inr: INRBatch, layer: PositionalEncoding):
    """positional encoding"""
    coords = inr.sampled_coords.unsqueeze(-1)
    n = 2**torch.arange(layer.N, device=coords.device) * 2*torch.pi * layer.scale
    embeddings = torch.cat((torch.sin(coords*n), torch.cos(coords*n)), dim=1).flatten(1)
    if layer.additive is True:
        return values + embeddings
    else:
        return torch.cat((values, embeddings), dim=-1)

def conv(values: torch.Tensor, # [B,N,c_in]
    inr: INRBatch, layer: nn.Module):
    """continuous convolution"""
    
    coords = inr.sampled_coords #[N,d]
    query_coords = _get_query_coords(inr, layer)

    if inr.sample_mode == 'grid' and hasattr(layer, 'shift'):
        query_coords = query_coords + layer.shift
    Diffs = query_coords.unsqueeze(1) - coords.unsqueeze(0)
    mask = layer.diffs_in_support(Diffs)
    padding_ratio = layer.kernel_intersection_ratio(query_coords)
    # if hasattr(layer, 'mask_tracker'):
    #     layer.mask_tracker = mask.sum(1).detach().cpu()

    # if layer.dropout > 0 and (inr.training and layer.training):
    #     mask *= torch.rand_like(mask, dtype=torch.half) > layer.dropout
    Y = values[:,torch.where(mask)[1]] # flattened list of values of neighborhood points
    Diffs = Diffs[mask] # flattened tensor of diffs between center coords and neighboring points
    lens = tuple(mask.sum(1)) # number of kernel points assigned to each point
    Ysplit = Y.split(lens, dim=1) # list of values at neighborhood points
    newVals = []

    if hasattr(layer, "interpolate_weights"):
        if inr.sample_mode == 'grid' or layer.N_bins != 0:
            ## group similar displacements
            bin_ixs, bin_centers = _cluster_diffs(Diffs, layer=layer, sample_mode=inr.sample_mode)

            if layer.groups != 1:
                if layer.groups == layer.out_channels and layer.groups == layer.in_channels:
                    w_o = layer.interpolate_weights(-bin_centers).squeeze(-1)
                    Wsplit = w_o.index_select(dim=0, index=bin_ixs).split(lens)
                    for ix,y in enumerate(Ysplit):
                        newVals.append(torch.einsum('bni,ni->bi',y,Wsplit[ix])/y.size(1))
                else:
                    raise NotImplementedError('groups')
                    # if g is num groups, each i/g channels produces o/g channels, then concat
                    w_og = layer.interpolate_weights(-bin_centers)
                    n,o,i_g = w_og.shape
                    g = layer.num_groups
                    o_g = o//g
                    Wsplit = w_og.view(n, o_g,g, i_g).index_select(dim=0, index=bin_ixs).split(lens)
                    for ix,y in enumerate(Ysplit):
                        newVals.append(torch.einsum('bnig,nogi->bog', y.reshape(-1, n, i_g, g),
                            Wsplit[ix]).flatten(1)/n)
            else:
                w_oi = layer.interpolate_weights(-bin_centers)
                Wsplit = w_oi.index_select(dim=0, index=bin_ixs).split(lens)
                for ix,y in enumerate(Ysplit):
                    newVals.append(torch.einsum('bni,noi->bo',y,Wsplit[ix])/y.size(1))
                
        else: ## calculate weights pairwise
            Dsplit = Diffs.split(lens) # list of diffs of neighborhood points
            if layer.groups != 1:
                if layer.groups == layer.out_channels and layer.groups == layer.in_channels:
                    for ix,y in enumerate(Ysplit):
                        w_o = layer.interpolate_weights(-Dsplit[ix]).squeeze(-1)
                        newVals.append(torch.einsum('bni,ni->bi',y,w_o)/y.size(1))
                else:
                    # if g is num groups, each i/g channels produces o/g channels, then concat
                    g = layer.groups
                    for ix,y in enumerate(Ysplit):
                        w_og = layer.interpolate_weights(-Dsplit[ix])
                        n,o,i_g = w_og.shape
                        o_g = o//g
                        newVals.append(torch.einsum('bnig,nogi->bog',
                            y.reshape(y.size(0), n, i_g, g), w_og.view(n, o_g, g, i_g)).flatten(1)/n)
            else:
                for ix,y in enumerate(Ysplit):
                    w_oi = layer.interpolate_weights(-Dsplit[ix])
                    newVals.append(torch.einsum('bni,noi->bo',y,w_oi)/y.size(1))
        
    else:
        if layer.groups != 1:
            raise NotImplementedError('convball')
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
            bin_centers = point_set.get_minNN_points_in_disk(radius=layer.radius, N=layer.N_bins)
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
    if padding_ratio is not None:
        newVals *= padding_ratio.unsqueeze(-1)

    if layer.bias is not None:
        newVals = newVals + layer.bias
    return newVals


def _cluster_diffs(x, layer, tol=.005, sample_mode=None):
    """Based on kmeans in https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html"""
    if sample_mode == 'grid' and hasattr(layer, 'grid_points'):
        c = layer.grid_points  # Initialize centroids to grid
    else:
        c = layer.sample_points # Initialize centroids with low-disc seq
    x_i = x.unsqueeze(1)  # (N, 1, D) samples
    c_j = c.unsqueeze(0)  # (1, K, D) centroids

    D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) squared distances
    minD, indices = D_ij.min(dim=1)
    cl = indices.view(-1)  # Points -> Nearest cluster
    if sample_mode == 'grid' and hasattr(layer, 'grid_points') and minD.mean() > tol:
        print("bad grid alignment")
        pdb.set_trace()

    return cl, c

def avg_pool(values: torch.Tensor, inr: INRBatch, layer: AvgPool,
    query_coords: Optional[PointSet]=None) -> torch.Tensor:
    """Average Pooling

    Args:
        values (torch.Tensor): _description_
        inr (INRBatch): _description_
        layer (AvgPool): _description_
        query_coords (Optional[PointSet], optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    pool_fxn = lambda x: x.mean(dim=2)
    return pool(pool_fxn, values, inr, layer, query_coords=query_coords)

def max_pool(values: torch.Tensor, inr: INRBatch, layer: MaxPool,
    query_coords: Optional[PointSet]=None):
    def pool_fxn(x):
        n = x.size(1)
        m = x.amax(1)
        if n == 1:
            return m
        return torch.where(m<0, m, m * (n+1)/(n-1) * 3/5)
    return pool(pool_fxn, values, inr, layer, query_coords=query_coords)

def pool(pool_fxn: Callable, values: torch.Tensor, inr: INRBatch, layer: MaxPool,
    query_coords: Optional[PointSet]=None):
    coords = inr.sampled_coords
    query_coords = _get_query_coords(inr, layer)

    if inr.sample_mode == 'grid' and hasattr(layer, 'shift'):
        query_coords = query_coords + layer.shift
    Diffs = query_coords.unsqueeze(1) - coords.unsqueeze(0)
    if hasattr(layer, 'num_neighbors'):
        mask = Diffs.norm(dim=-1).topk(k=layer.num_neighbors, dim=1, largest=False).indices
        return pool_fxn(values[:,mask]) 
    else:
        mask = layer.diffs_in_support(Diffs)

    Y = values[:,torch.where(mask)[1]]
    return torch.stack([pool_fxn(y) for y in Y.split(tuple(mask.sum(1)), dim=1)], dim=1)

    

def max_pool_kernel(values, inr, layer, query_coords=None):
    coords = inr.sampled_coords
    query_coords = _get_query_coords(inr, layer)

    if hasattr(layer, "norm"):
        Diffs = query_coords.unsqueeze(1) - coords.unsqueeze(0)
        mask = layer.norm(Diffs) < layer.radius
    else:
        if torch.amax(layer.shift) > 0:
            query_coords = query_coords + layer.shift
        Diffs = query_coords.unsqueeze(1) - coords.unsqueeze(0)
        mask = layer.diffs_in_support(Diffs)
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

def _get_query_coords(inr, layer):
    coords = inr.sampled_coords
    if layer.down_ratio != 1 and layer.down_ratio != 0:
        if layer.down_ratio > 1: 
            layer.down_ratio = 1/layer.down_ratio
        N = round(coords.size(0)*layer.down_ratio)
        if inr.sample_mode != 'qmc':
            if not hasattr(inr, 'dropped_coords'):
                inr.dropped_coords = coords[N:]
            else:
                inr.dropped_coords = torch.cat((coords[N:], inr.dropped_coords), dim=0)
        inr.sampled_coords = query_coords = coords[:N]
    else:
        query_coords = coords
    return query_coords


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

