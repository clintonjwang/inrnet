"""Quasi-Monte Carlo Method"""

# class QMCIntegrator:
#     def __init__(self, integrand):
#         self.integrand = integrand
#     def __call__(self, point_set):
#         return self.integrand(point_set.values).mean()

# def get_ball_volume(r, dims):
#     return ((2.*math.pi**(dims/2.))/(dims*math.gamma(dims/2.)))*r**dims

# def subsample_points_by_grid(coords, spacing, input_dims=2, random=False):
#     x = coords[...,0] / spacing[0]
#     y = coords[...,1] / spacing[1]
#     x -= x.min()
#     y -= y.min()
#     bin_ixs = torch.floor(torch.stack((x,y), dim=-1) + 1e-4).int()
#     bin_ixs = bin_ixs[:,0]*int(3/spacing[1]) + bin_ixs[:,1] # TODO: adapt for larger domains, d
#     bins = torch.unique(bin_ixs)
#     matches = bin_ixs.unsqueeze(0) == bins.unsqueeze(1) # (bin, point)
#     points_per_bin = tuple(matches.sum(1))
#     if random:
#         surviving_indices = [x[np.random.randint(0,len(x))] for x in torch.where(matches)[1].split(points_per_bin)]
#     else:
#         surviving_indices = [x[0] for x in torch.where(matches)[1].split(points_per_bin)]
#         # def select_topleft(indices):
#         #     return indices[torch.min(coords[indices,0] + coords[indices,1]*.1, dim=0).indices.item()]
#         # surviving_indices = [select_topleft(x) for x in torch.where(matches)[1].split(points_per_bin)]

#     return coords[surviving_indices,:]


# def interpolate(query_coords, observed_coords, values):
#     if query_coords.size(0) == 0:
#         return values.new_zeros(0, values.size(1))

#     dists = (query_coords.unsqueeze(0) - observed_coords.unsqueeze(1)).norm(dim=-1)
#     r, indices = dists.topk(3, dim=0, largest=False)
#     q = (1/r).unsqueeze(-1)
#     Q = q.sum(0)
#     sv = (values[indices[0]] * q[0] + values[indices[1]] * q[1] + values[indices[2]] * q[2])/Q
#     return sv

# def interpolate_weights_single_channel(xy, tx,ty,c, order=2):
#     W = []
#     X = xy[:,0].unsqueeze(1)
#     Y = xy[:,1].unsqueeze(1)
#     px = py = order

#     values, kx = (tx<=X).min(dim=-1)
#     values, ky = (ty<=Y).min(dim=-1)
#     kx -= 1
#     ky -= 1
#     kx[values] = tx.size(-1)-px-2
#     ky[values] = ty.size(-1)-py-2

#     for z in range(X.size(0)):
#         D = c[kx[z]-px : kx[z]+1, ky[z]-py : ky[z]+1].clone()

#         for r in range(1, px + 1):
#             try:
#                 alphax = (X[z,0] - tx[kx[z]-px+1:kx[z]+1]) / (
#                     tx[2+kx[z]-r:2+kx[z]-r+px] - tx[kx[z]-px+1:kx[z]+1])
#             except RuntimeError:
#                 print("input off the grid")
#                 pdb.set_trace()
#             for j in range(px, r - 1, -1):
#                 D[j] = (1-alphax[j-1]) * D[j-1] + alphax[j-1] * D[j].clone()

#         for r in range(1, py + 1):
#             alphay = (Y[z,0] - ty[ky[z]-py+1:ky[z]+1]) / (
#                 ty[2+ky[z]-r:2+ky[z]-r+py] - ty[ky[z]-py+1:ky[z]+1])
#             for j in range(py, r-1, -1):
#                 D[px,j] = (1-alphay[j-1]) * D[px,j-1].clone() + alphay[j-1] * D[px,j].clone()
        
#         W.append(D[px,py])
#     return torch.stack(w_oi)

