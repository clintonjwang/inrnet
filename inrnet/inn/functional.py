"""Functions"""
import pdb
from typing import Callable, Optional
import torch
Tensor = torch.Tensor

from inrnet.inn.inr import INRBatch
from inrnet.inn.layers.other import PositionalEncoding
from inrnet.inn.point_set import PointSet, PointValues
from inrnet.inn.support import Support
nn=torch.nn

def change_sample_density(values: PointValues, inr: INRBatch):
    coords = inr.sampled_coords
    return coords

def tokenization(values: PointValues, inr: INRBatch):
    """tokenization"""
    return values
    
def pos_enc(values: PointValues, inr: INRBatch, N: int,
    scale: float=1., additive: bool=True):
    """positional encoding"""
    coords = inr.sampled_coords.unsqueeze(-1)
    n = 2**torch.arange(N, device=coords.device) * 2*torch.pi * scale
    embeddings = torch.cat((torch.sin(coords*n), torch.cos(coords*n)), dim=1).flatten(1)
    if additive is True:
        return values + embeddings
    else:
        return torch.cat((values, embeddings), dim=-1)

def conv(values: PointValues, # [B,N,c_in]
    inr: INRBatch, out_channels: int,
    coord_to_weights: Callable[[PointSet], Tensor],
    support: Support, down_ratio: float,
    N_bins: int=0, groups: int=1,
    grid_points=None, qmc_points=None,
    bias: Tensor|None=None) -> PointValues:
    """Continuous convolution

    Args:
        values (PointValues): _description_
        inr (INRBatch): input INR
        out_channels (int): _description_
        coord_to_weights (Callable[[PointSet], Tensor]): _description_
        support (Support): _description_
        down_ratio (float): _description_
        N_bins (int, optional): _description_. Defaults to 0.
        groups (int, optional): _description_. Defaults to 1.
        grid_points (_type_, optional): _description_. Defaults to None.
        qmc_points (_type_, optional): _description_. Defaults to None.
        bias (Tensor | None, optional): _description_. Defaults to None.

    Returns:
        PointValues: _description_
    """
    
    coords = inr.sampled_coords #[N,d]
    query_coords = _get_query_coords(inr, down_ratio)
    in_channels = inr.channels

    if inr.sample_mode == 'grid' and hasattr(support, 'grid_shift'):
        query_coords = query_coords + support.grid_shift
    Diffs = query_coords.unsqueeze(1) - coords.unsqueeze(0)
    mask = support.in_support(Diffs)
    padding_ratio = support.kernel_intersection_ratio(query_coords)
    # if hasattr(layer, 'mask_tracker'):
    #     layer.mask_tracker = mask.sum(1).detach().cpu()

    # if layer.dropout > 0 and (inr.training and layer.training):
    #     mask *= torch.rand_like(mask, dtype=torch.half) > layer.dropout
    Y = values[:,torch.where(mask)[1]] # flattened list of values of neighborhood points
    Diffs = Diffs[mask] # flattened tensor of diffs between center coords and neighboring points
    lens = tuple(mask.sum(1)) # number of kernel points assigned to each point
    Ysplit = Y.split(lens, dim=1) # list of values at neighborhood points
    newVals = []

    if inr.sample_mode == 'grid' or N_bins != 0:
        ## group similar displacements
        bin_ixs, bin_centers = _cluster_points(Diffs, grid_points=grid_points, qmc_points=qmc_points,
            sample_mode=inr.sample_mode)

        if groups != 1:
            if groups == out_channels and groups == in_channels:
                w_o = coord_to_weights(-bin_centers).squeeze(-1)
                Wsplit = w_o.index_select(dim=0, index=bin_ixs).split(lens)
                for ix,y in enumerate(Ysplit):
                    newVals.append(torch.einsum('bni,ni->bi',y,Wsplit[ix])/y.size(1))
            else:
                # if g is num groups, each i/g channels produces o/g channels, then concat
                w_og = coord_to_weights(-bin_centers)
                n,o,i_g = w_og.shape
                g = groups
                o_g = o//g
                Wsplit = w_og.view(n, o_g, g, i_g).index_select(dim=0, index=bin_ixs).split(lens)
                for ix,y in enumerate(Ysplit):
                    newVals.append(torch.einsum('bnig,nogi->bog', y.reshape(-1, n, i_g, g),
                        Wsplit[ix]).flatten(1)/n)
        else:
            w_oi = coord_to_weights(-bin_centers)
            Wsplit = w_oi.index_select(dim=0, index=bin_ixs).split(lens)
            for ix,y in enumerate(Ysplit):
                newVals.append(torch.einsum('bni,noi->bo',y,Wsplit[ix])/y.size(1))
            
    else: ## calculate weights pairwise
        Dsplit = Diffs.split(lens) # list of diffs of neighborhood points
        if groups != 1:
            if groups == out_channels and groups == in_channels:
                for ix,y in enumerate(Ysplit):
                    w_o = coord_to_weights(-Dsplit[ix]).squeeze(-1)
                    newVals.append(torch.einsum('bni,ni->bi',y,w_o)/y.size(1))
            else:
                # if g is num groups, each i/g channels produces o/g channels, then concat
                g = groups
                for ix,y in enumerate(Ysplit):
                    w_og = coord_to_weights(-Dsplit[ix])
                    n,o,i_g = w_og.shape
                    o_g = o//g
                    newVals.append(torch.einsum('bnig,nogi->bog',
                        y.reshape(y.size(0), n, i_g, g), w_og.view(n, o_g, g, i_g)).flatten(1)/n)
        else:
            for ix,y in enumerate(Ysplit):
                w_oi = coord_to_weights(-Dsplit[ix])
                newVals.append(torch.einsum('bni,noi->bo',y,w_oi)/y.size(1))
                # if y.size(1) == 0:
                #     newVals.append(y.new_zeros(y.size(0), layer.out_channels))
                # else:
                #     newVals.append(y.unsqueeze(1).matmul(w_oi).squeeze(1).mean(0))
        
    newVals = torch.stack(newVals, dim=1) #[B,N,c_out]
    if padding_ratio is not None:
        newVals *= padding_ratio.unsqueeze(-1)

    if bias is not None:
        newVals = newVals + bias
    return newVals.as_subclass(PointValues)


def _cluster_points(points: PointSet,
    grid_points:PointSet|None=None,
    qmc_points:PointSet|None=None, sample_mode=None, tol=.005):
    """Cluster a point set

    Args:
        points (PointSet): points to cluster
        grid_points (PointSet | None, optional): cluster centers on the grid. Defaults to None.
        qmc_points (PointSet | None, optional): cluster centers from QMC. Defaults to None.
        sample_mode (_type_, optional): _description_. Defaults to None.
        tol (float, optional): _description_. Defaults to .005.

    Returns:
        _type_: _description_
    """    
    """Based on kmeans in https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html"""
    if sample_mode == 'grid' and grid_points is not None:
        c = grid_points  # Initialize centroids to grid
    else:
        c = qmc_points # Initialize centroids with low-disc seq
    x_i = points.unsqueeze(1)  # (N, 1, D) samples
    c_j = c.unsqueeze(0)  # (1, K, D) centroids

    D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) squared distances
    minD, indices = D_ij.min(dim=1)
    cl = indices.view(-1)  # Points -> Nearest cluster
    if sample_mode == 'grid' and grid_points is not None and minD.mean() > tol:
        raise ValueError("bad grid alignment")

    return cl, c

def avg_pool(values: PointValues, inr: INRBatch, support: Support, down_ratio: float) -> PointValues:
    """Average Pooling

    Args:
        values (Tensor): _description_
        inr (INRBatch): _description_
        layer (AvgPool): _description_
        query_coords (Optional[PointSet], optional): _description_. Defaults to None.

    Returns:
        Tensor: _description_
    """
    pool_fxn = lambda x: x.mean(dim=2)
    return pool_kernel(pool_fxn, values, inr, support, down_ratio)

def max_pool(values: PointValues, inr: INRBatch, support: Support, down_ratio: float) -> PointValues:
    """Max Pooling

    Returns:
        PointValues: pooled values
    """    
    def pool_fxn(x):
        n = x.size(1)
        m = x.amax(1)
        if n == 1:
            return m
        return torch.where(m<0, m, m * (n+1)/(n-1) * 3/5)
    return pool_kernel(pool_fxn, values, inr, support, down_ratio)

def pool_kernel(pool_fxn: Callable, values: Tensor, inr: INRBatch,
        support: Support, down_ratio: float):
    coords = inr.sampled_coords
    query_coords = _get_query_coords(inr, down_ratio)

    if inr.sample_mode == 'grid' and hasattr(support, 'center'):
        query_coords = query_coords + support.center.to(device=query_coords.device)
    Diffs = query_coords.unsqueeze(1) - coords.unsqueeze(0)
    mask = support.in_support(Diffs)
    Y = values[:,torch.where(mask)[1]]
    return torch.stack([pool_fxn(y) for y in Y.split(tuple(mask.sum(1)), dim=1)], dim=1)


def pool_neighbors(pool_fxn: Callable, values: Tensor, inr: INRBatch,
        num_neighbors: int, down_ratio: float, shift: Tensor):
    coords = inr.sampled_coords
    query_coords = _get_query_coords(inr, down_ratio)

    if inr.sample_mode == 'grid' and shift:
        query_coords = query_coords + shift.to(device=query_coords.device)
    Diffs = query_coords.unsqueeze(1) - coords.unsqueeze(0)
    mask = Diffs.norm(dim=-1).topk(k=num_neighbors, dim=1, largest=False).indices
    return pool_fxn(values[:,mask])


### Integrations over I

def inner_product(values: PointValues, inr: INRBatch, layer) -> Tensor:
    W_ij = layer.m_ij(inr.sampled_coords) #(B,d) -> (B,cin,cout)
    if layer.normalized:
        out = values.unsqueeze(1).matmul(torch.softmax(W_ij, dim=-1)).squeeze(1)
    else:
        out = values.unsqueeze(1).matmul(W_ij).squeeze(1)
    if hasattr(layer, "b_j"):
        out = out + layer.b_j
    return out

# def global_avg_pool(values: PointValues) -> Tensor:
#     """Collapses spatial coordinate (not in use)

#     Args:
#         values (PointValues): has shape (N,d) or (B,N,d)

#     Returns:
#         Tensor: has shape (B,1,d)
#     """
#     return values.mean(-2, keepdim=True)

# def adaptive_avg_pool(values: PointValues, inr: INRBatch, layer) -> Tensor:
#     coords = inr.sampled_coords
#     Diffs = coords.unsqueeze(0) - coords.unsqueeze(1)
#     mask = layer.norm(Diffs) < layer.radius 
#     Y = values[torch.where(mask)[1]]
#     return torch.stack([y.mean(0) for y in Y.split(tuple(mask.sum(0)))])

def _get_query_coords(inr: INRBatch, down_ratio: float) -> PointSet:
    """Get coordinates for the output INR

    Args:
        inr (INRBatch): input INR
        down_ratio (float): ratio between number of output and input points

    Returns:
        PointSet: coordinates of output INR
    """    
    coords = inr.sampled_coords
    if down_ratio != 1 and down_ratio != 0:
        if down_ratio > 1: 
            down_ratio = 1/down_ratio
        N = round(coords.size(0)*down_ratio)
        if inr.sample_mode != 'qmc':
            if not hasattr(inr, 'dropped_coords'):
                inr.dropped_coords = coords[N:]
            else:
                inr.dropped_coords = torch.cat((coords[N:], inr.dropped_coords), dim=0)
        inr.sampled_coords = query_coords = coords[:N]
    else:
        query_coords = coords
    return query_coords


def normalize(values: PointValues, mean, var, eps=1e-5):
    return (values - mean)/(var.sqrt() + eps)
