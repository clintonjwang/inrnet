"""Functions"""
from __future__ import annotations
import pdb
from typing import TYPE_CHECKING

from inrnet.inn.point_set import PointSet, PointValues
if TYPE_CHECKING:
    from inrnet.inn.inr import DiscretizedINR
    from inrnet.inn.support import Support
from typing import Callable
import torch
Tensor = torch.Tensor
nn=torch.nn

# def change_sample_density(inr: DiscretizedINR):
#     coords = inr.coords
#     return coords

def tokenization(inr: DiscretizedINR):
    """tokenization"""
    return inr
    
def pos_enc(inr: DiscretizedINR, N: int,
    scale: float=1., additive: bool=True):
    """positional encoding"""
    coords = inr.coords.unsqueeze(-1)
    n = 2**torch.arange(N, device=coords.device) * 2*torch.pi * scale
    embeddings = torch.cat((torch.sin(coords*n), torch.cos(coords*n)), dim=1).flatten(1)
    if additive is True:
        inr.values = inr.values + embeddings
    else:
        inr.values = torch.cat((inr.values, embeddings), dim=-1)
    return inr

def conv(inr: DiscretizedINR, out_channels: int,
    coord_to_weights: Callable[[PointSet], Tensor],
    kernel_support: Support, down_ratio: float,
    N_bins: int=0, groups: int=1,
    grid_points=None, qmc_points=None,
    bias: Tensor|None=None) -> PointValues:
    """Continuous convolution

    Args:
        inr (DiscretizedINR): input INR
        out_channels (int): _description_
        coord_to_weights (Callable[[PointSet], Tensor]): _description_
        kernel_support (Support): _description_
        down_ratio (float): _description_
        N_bins (int, optional): _description_. Defaults to 0.
        groups (int, optional): _description_. Defaults to 1.
        grid_points (_type_, optional): _description_. Defaults to None.
        qmc_points (_type_, optional): _description_. Defaults to None.
        bias (Tensor | None, optional): _description_. Defaults to None.

    Returns:
        PointValues: _description_
    """
    
    query_coords = _get_query_coords(inr, down_ratio)
    in_channels = inr.channels

    Diffs = query_coords.unsqueeze(1) - inr.coords.unsqueeze(0)
    mask = kernel_support.in_support(Diffs)
    # padding_ratio = kernel_support.kernel_intersection_ratio(query_coords)
    # if hasattr(layer, 'mask_tracker'):
    #     layer.mask_tracker = mask.sum(1).detach().cpu()

    # if layer.dropout > 0 and (inr.training and layer.training):
    #     mask *= torch.rand_like(mask, dtype=torch.half) > layer.dropout
    Y = inr.values[:,torch.where(mask)[1]] # flattened list of values of neighborhood points
    Diffs = Diffs[mask] # flattened tensor of diffs between center coords and neighboring points
    lens = tuple(mask.sum(1)) # number of kernel points assigned to each point
    Ysplit = Y.split(lens, dim=1) # list of values at neighborhood points
    newVals = []

    if inr.coords.sample_mode == 'grid' or N_bins != 0:
        ## group similar displacements
        bin_ixs, bin_centers = _cluster_points(Diffs, grid_points=grid_points, qmc_points=qmc_points,
            sample_mode=inr.coords.sample_mode)

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
    # if padding_ratio is not None:
    #     newVals *= padding_ratio.unsqueeze(-1)

    if bias is not None:
        newVals = newVals + bias
    inr.values = newVals#.as_subclass(PointValues)
    inr.coords = query_coords
    return inr


def _cluster_points(points: PointSet,
    grid_points:PointSet|None=None,
    qmc_points:PointSet|None=None, sample_mode=None, tol=.005):
    """Cluster a point set
    Based on kmeans in https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html
    
    Args:
        points (PointSet): points to cluster
        grid_points (PointSet | None, optional): cluster centers on the grid. Defaults to None.
        qmc_points (PointSet | None, optional): cluster centers from QMC. Defaults to None.
        sample_mode (_type_, optional): _description_. Defaults to None.
        tol (float, optional): _description_. Defaults to .005.

    Returns:
        _type_: _description_
    """
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

def avg_pool(inr: DiscretizedINR, kernel_support: Support, down_ratio: float) -> PointValues:
    pool_fxn = lambda x: x.mean(dim=2)
    return pool_kernel(pool_fxn, inr, kernel_support, down_ratio)

def max_pool(inr: DiscretizedINR, kernel_support: Support, down_ratio: float) -> PointValues:
    def pool_fxn(x):
        n = x.size(1)
        m = x.amax(1)
        if n == 1:
            return m
        return torch.where(m<0, m, m * (n+1)/(n-1) * 3/5)
    return pool_kernel(pool_fxn, inr, kernel_support, down_ratio)

def pool_kernel(pool_fxn: Callable, inr: DiscretizedINR,
        kernel_support: Support, down_ratio: float):
    query_coords = _get_query_coords(inr, down_ratio)
    Diffs = query_coords.unsqueeze(1) - inr.coords.unsqueeze(0)
    mask = kernel_support.in_support(Diffs)
    Y = inr.values[:,torch.where(mask)[1]]
    return torch.stack([pool_fxn(y) for y in Y.split(tuple(mask.sum(1)), dim=1)], dim=1)


### Integrations over I

def _get_query_coords(inr: DiscretizedINR, down_ratio: float) -> PointSet:
    """Get coordinates for the output INR

    Args:
        inr (DiscretizedINR): input INR
        down_ratio (float): ratio between number of output and input points

    Returns:
        PointSet: coordinates of output INR
    """    
    if down_ratio != 1 and down_ratio != 0:
        if down_ratio > 1: 
            down_ratio = 1/down_ratio
        N = round(inr.coords.size(0)*down_ratio)
        # if inr.sample_mode != 'qmc':
        #     if not hasattr(inr, 'dropped_coords'):
        #         inr.dropped_coords = inr.coords[N:]
        #     else:
        #         inr.dropped_coords = torch.cat((inr.coords[N:], inr.dropped_coords), dim=0)
        query_coords = inr.coords[:N]
        query_coords.sample_mode = inr.coords.sample_mode
    else:
        query_coords = inr.coords
    return query_coords


def normalize(inr: DiscretizedINR, mean, var, eps=1e-5):
    inr.values = (inr.values - mean)/(var.sqrt() + eps)
    return inr
