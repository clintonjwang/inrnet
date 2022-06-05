# import numpy as np
# import torch
# import torch.nn as nn
# from torch import distributions as dist

# class DVR(nn.Module):
#     ''' DVR model class.

#     Args:
#         decoder (nn.Module): decoder network
#         encoder (nn.Module): encoder network
#         device (device): torch device
#         depth_function_kwargs (dict): keyworded arguments for the
#             depth_function
#     '''

#     def __init__(self, decoder, device=None, depth_function_kwargs={}):
#         super().__init__()
#         self.decoder = decoder.to(device)

#         self._device = device
#         self.call_depth_function = DepthModule(
#             **depth_function_kwargs)

#     def get_normals(self, points, mask, c=None, h_sample=1e-3,
#                     h_finite_difference=1e-3):
#         ''' Returns the unit-length normals for points and one randomly
#         sampled neighboring point for each point.

#         Args:
#             points (tensor): points tensor
#             mask (tensor): mask for points
#             c (tensor): latent conditioned code c
#             h_sample (float): interval length for sampling the neighbors
#             h_finite_difference (float): step size finite difference-based
#                 gradient calculations
#         '''
#         device = self._device

#         if mask.sum() > 0:
#             c = c.unsqueeze(1).repeat(1, points.shape[1], 1)[mask]
#             points = points[mask]
#             points_neighbor = points + (torch.rand_like(points) * h_sample -
#                                         (h_sample / 2.))

#             normals_p = normalize_tensor(
#                 self.get_central_difference(points, c=c,
#                                             h=h_finite_difference))
#             normals_neighbor = normalize_tensor(
#                 self.get_central_difference(points_neighbor, c=c,
#                                             h=h_finite_difference))
#         else:
#             normals_p = torch.empty(0, 3).to(device)
#             normals_neighbor = torch.empty(0, 3).to(device)

#         return [normals_p, normals_neighbor]

#     def get_central_difference(self, points, c=None, h=1e-3):
#         ''' Calculates the central difference for points.

#         It approximates the derivative at the given points as follows:
#             f'(x) ≈ f(x + h/2) - f(x - h/2) for a small step size h

#         Args:
#             points (tensor): points
#             c (tensor): latent conditioned code c
#             h (float): step size for central difference method
#         '''
#         n_points, _ = points.shape
#         device = self._device

#         if c.shape[-1] != 0:
#             c = c.unsqueeze(1).repeat(1, 6, 1).view(-1, c.shape[-1])

#         # calculate steps x + h/2 and x - h/2 for all 3 dimensions
#         step = torch.cat([
#             torch.tensor([1., 0, 0]).view(1, 1, 3).repeat(n_points, 1, 1),
#             torch.tensor([-1., 0, 0]).view(1, 1, 3).repeat(n_points, 1, 1),
#             torch.tensor([0, 1., 0]).view(1, 1, 3).repeat(n_points, 1, 1),
#             torch.tensor([0, -1., 0]).view(1, 1, 3).repeat(n_points, 1, 1),
#             torch.tensor([0, 0, 1.]).view(1, 1, 3).repeat(n_points, 1, 1),
#             torch.tensor([0, 0, -1.]).view(1, 1, 3).repeat(n_points, 1, 1)
#         ], dim=1).to(device) * h / 2
#         points_eval = (points.unsqueeze(1).repeat(1, 6, 1) + step).view(-1, 3)

#         # Eval decoder at these points
#         f = self.decoder(points_eval, c=c, only_occupancy=True,
#                          batchwise=False).view(n_points, 6)

#         # Get approximate derivate as f(x + h/2) - f(x - h/2)
#         df_dx = torch.stack([
#             (f[:, 0] - f[:, 1]),
#             (f[:, 2] - f[:, 3]),
#             (f[:, 4] - f[:, 5]),
#         ], dim=-1)
#         return df_dx

#     def decode(self, p, c=None, **kwargs):
#         ''' Returns occupancy probabilities for the sampled points.

#         Args:
#             p (tensor): points
#             c (tensor): latent conditioned code c
#         '''

#         logits = self.decoder(p, c, only_occupancy=True, **kwargs)
#         p_r = dist.Bernoulli(logits=logits)
#         return p_r

#     def march_along_ray(self, ray0, ray_direction, c=None, it=None,
#                         sampling_accuracy=None):
#         ''' Marches along the ray and returns the d_i values in the formula
#             r(d_i) = ray0 + ray_direction * d_i
#         which returns the surfaces points.

#         Here, ray0 and ray_direction are directly used without any
#         transformation; Hence the evaluation is done in object-centric
#         coordinates.

#         Args:
#             ray0 (tensor): ray start points (camera centers)
#             ray_direction (tensor): direction of rays; these should be the
#                 vectors pointing towards the pixels
#             c (tensor): latent conditioned code c
#             it (int): training iteration (used for ray sampling scheduler)
#             sampling_accuracy (tuple): if not None, this overwrites the default
#                 sampling accuracy ([128, 129])
#         '''
#         device = self._device

#         d_i = self.call_depth_function(ray0, ray_direction, self.decoder,
#                                        c=c, it=it, n_steps=sampling_accuracy)

#         # Get mask for where first evaluation point is occupied
#         mask_zero_occupied = d_i == 0

#         # Get mask for predicted depth
#         mask_pred = get_mask(d_i).detach()

#         # For sanity for the gradients
#         d_hat = torch.ones_like(d_i).to(device)
#         d_hat[mask_pred] = d_i[mask_pred]
#         d_hat[mask_zero_occupied] = 0.

#         return d_hat, mask_pred, mask_zero_occupied

#     def pixels_to_world(self, pixels, camera_mat, world_mat, scale_mat, c,
#                         it=None, sampling_accuracy=None):
#         ''' Projects pixels to the world coordinate system.

#         Args:
#             pixels (tensor): sampled pixels in range [-1, 1]
#             camera_mat (tensor): camera matrices
#             world_mat (tensor): world matrices
#             scale_mat (tensor): scale matrices
#             c (tensor): latent conditioned code c
#             it (int): training iteration (used for ray sampling scheduler)
#             sampling_accuracy (tuple): if not None, this overwrites the default
#                 sampling accuracy ([128, 129])
#         '''
#         batch_size, n_points, _ = pixels.shape
#         pixels_world = image_points_to_world(pixels, camera_mat, world_mat,
#                                              scale_mat)
#         camera_world = origin_to_world(n_points, camera_mat, world_mat,
#                                        scale_mat)
#         ray_vector = (pixels_world - camera_world)

#         d_hat, mask_pred, mask_zero_occupied = self.march_along_ray(
#             camera_world, ray_vector, c, it, sampling_accuracy)
#         p_world_hat = camera_world + ray_vector * d_hat.unsqueeze(-1)
#         return p_world_hat, mask_pred, mask_zero_occupied

#     def decode_color(self, p_world, c=None, **kwargs):
#         ''' Decodes the color values for world points.

#         Args:
#             p_world (tensor): world point tensor
#             c (tensor): latent conditioned code c
#         '''
#         rgb_hat = self.decoder(p_world, c=c, only_texture=True)
#         rgb_hat = torch.sigmoid(rgb_hat)
#         return rgb_hat

#     def to(self, device):
#         ''' Puts the model to the device.

#         Args:
#             device (device): pytorch device
#         '''
#         model = super().to(device)
#         model._device = device
#         return model




# import torch.nn as nn
# import torch.nn.functional as F


# class Decoder(nn.Module):
#     ''' Decoder class.

#     As discussed in the paper, we implement the OccupancyNetwork
#     f and TextureField t in a single network. It consists of 5
#     fully-connected ResNet blocks with ReLU activation.

#     Args:
#         dim (int): input dimension
#         z_dim (int): dimension of latent code z
#         c_dim (int): dimension of latent conditioned code c
#         hidden_size (int): hidden size of Decoder network
#         leaky (bool): whether to use leaky ReLUs
#         n_blocks (int): number of ResNet blocks
#         out_dim (int): output dimension (e.g. 1 for only
#             occupancy prediction or 4 for occupancy and
#             RGB prediction)
#     '''

#     def __init__(self, dim=3, c_dim=128,
#                  hidden_size=512, leaky=False, n_blocks=5, out_dim=4):
#         super().__init__()
#         self.c_dim = c_dim
#         self.n_blocks = n_blocks
#         self.out_dim = out_dim

#         # Submodules
#         self.fc_p = nn.Linear(dim, hidden_size)
#         self.fc_out = nn.Linear(hidden_size, out_dim)

#         if c_dim != 0:
#             self.fc_c = nn.ModuleList([
#                 nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
#             ])

#         self.blocks = nn.ModuleList([
#             ResnetBlockFC(hidden_size) for i in range(n_blocks)
#         ])

#         if not leaky:
#             self.actvn = F.relu
#         else:
#             self.actvn = lambda x: F.leaky_relu(x, 0.2)

#     def forward(self, p, c=None, batchwise=True, only_occupancy=False,
#                 only_texture=False, **kwargs):

#         assert((len(p.shape) == 3) or (len(p.shape) == 2))

#         net = self.fc_p(p)
#         for n in range(self.n_blocks):
#             if self.c_dim != 0 and c is not None:
#                 net_c = self.fc_c[n](c)
#                 if batchwise:
#                     net_c = net_c.unsqueeze(1)
#                 net = net + net_c

#             net = self.blocks[n](net)

#         out = self.fc_out(self.actvn(net))

#         if only_occupancy:
#             if len(p.shape) == 3:
#                 out = out[:, :, 0]
#             elif len(p.shape) == 2:
#                 out = out[:, 0]
#         elif only_texture:
#             if len(p.shape) == 3:
#                 out = out[:, :, 1:4]
#             elif len(p.shape) == 2:
#                 out = out[:, 1:4]

#         out = out.squeeze(-1)
#         return out

# class ResnetBlockFC(nn.Module):
#     ''' Fully connected ResNet Block class.

#     Args:
#         size_in (int): input dimension
#         size_out (int): output dimension
#         size_h (int): hidden dimension
#     '''

#     def __init__(self, size_in, size_out=None, size_h=None):
#         super().__init__()
#         # Attributes
#         if size_out is None:
#             size_out = size_in

#         if size_h is None:
#             size_h = min(size_in, size_out)

#         self.size_in = size_in
#         self.size_h = size_h
#         self.size_out = size_out
#         # Submodules
#         self.fc_0 = nn.Linear(size_in, size_h)
#         self.fc_1 = nn.Linear(size_h, size_out)
#         self.actvn = nn.ReLU()

#         if size_in == size_out:
#             self.shortcut = None
#         else:
#             self.shortcut = nn.Linear(size_in, size_out, bias=False)
#         # Initialization
#         nn.init.zeros_(self.fc_1.weight)

#     def forward(self, x):
#         net = self.fc_0(self.actvn(x))
#         dx = self.fc_1(self.actvn(net))

#         if self.shortcut is not None:
#             x_s = self.shortcut(x)
#         else:
#             x_s = x

#         return x_s + dx


# class DepthModule(nn.Module):
#     ''' Depth Module class.

#     The depth module is a wrapper class for the autograd function
#     DepthFunction (see below).

#     Args:
#         tau (float): threshold value
#         n_steps (tuple): number of evaluation steps; if the difference between
#             n_steps[0] and n_steps[1] is larger then 1, the value is sampled
#             in the range
#         n_secant_steps (int): number of secant refinement steps
#         depth_range (tuple): range of possible depth values; not relevant when
#             unit cube intersection is used
#         method (string): refinement method (default: 'scant')
#         check_cube_intersection (bool): whether to intersect rays with unit
#             cube for evaluations
#         max_points (int): max number of points loaded to GPU memory
#         schedule_ray_sampling (bool): whether to schedule ray sampling accuracy
#         scheduler_milestones (list): list of scheduler milestones after which
#             the accuracy is doubled. This overwrites n_steps if chosen.
#         init_resolution (int): initial resolution
#     '''

#     def __init__(self, tau=0.5, n_steps=[128, 129], n_secant_steps=8,
#                  depth_range=[0., 2.4], method='secant',
#                  check_cube_intersection=True, max_points=3700000,
#                  schedule_ray_sampling=True,
#                  schedule_milestones=[50000, 100000, 250000],
#                  init_resolution=16):
#         super().__init__()
#         self.tau = tau
#         self.n_steps = n_steps
#         self.n_secant_steps = n_secant_steps
#         self.depth_range = depth_range
#         self.method = method
#         self.check_cube_intersection = check_cube_intersection
#         self.max_points = max_points
#         self.schedule_ray_sampling = schedule_ray_sampling

#         self.schedule_milestones = schedule_milestones
#         self.init_resolution = init_resolution

#         self.calc_depth = DepthFunction.apply

#     def get_sampling_accuracy(self, it):
#         ''' Returns sampling accuracy for current training iteration.

#         Args:
#             it (int): training iteration
#         '''
#         if len(self.schedule_milestones) == 0:
#             return [128, 129]
#         else:
#             res = self.init_resolution
#             for i, milestone in enumerate(self.schedule_milestones):
#                 if it < milestone:
#                     return [res, res + 1]
#                 res = res * 2
#             return [res, res + 1]

#     def forward(self, ray0, ray_direction, decoder, c=None, it=None,
#                 n_steps=None):
#         ''' Calls the depth function and returns predicted depth values.

#         NOTE: To avoid transformations, we assume to already have world
#         coordinates and we return the d_i values of the function
#             ray(d_i) = ray0 + d_i * ray_direction
#         for ease of computation.
#         (We can later transform the predicted points e.g. to the camera space
#         to obtain the "normal" depth value as the z-axis of the transformed
#         point.)

#         Args:
#             ray0 (tensor): ray starting points (camera center)
#             ray_direction (tensor): direction of ray
#             decoder (nn.Module): decoder model to evaluate points on the ray
#             c (tensor): latent conditioned code c
#             it (int): training iteration (used for ray sampling scheduler)
#             n_steps (tuple): number of evaluation steps; this overwrites
#                 self.n_steps if not None.
#         '''
#         device = ray0.device
#         batch_size, n_p, _ = ray0.shape
#         if n_steps is None:
#             if self.schedule_ray_sampling and it is not None:
#                 n_steps = self.get_sampling_accuracy(it)
#             else:
#                 n_steps = self.n_steps
#         if n_steps[1] > 1:
#             inputs = [ray0, ray_direction, decoder, c, n_steps,
#                       self.n_secant_steps, self.tau, self.depth_range,
#                       self.method, self.check_cube_intersection,
#                       self.max_points] + [k for k in decoder.parameters()]
#             d_hat = self.calc_depth(*inputs)
#         else:
#             d_hat = torch.full((batch_size, n_p), np.inf).to(device)
#         return d_hat


# class DepthFunction(torch.autograd.Function):
#     ''' Depth Function class.

#     It provides the function to march along given rays to detect the surface
#     points for the OccupancyNetwork. The backward pass is implemented using
#     the analytic gradient described in the publication.
#     '''
#     @staticmethod
#     def run_Bisection_method(d_low, d_high, n_secant_steps, ray0_masked,
#                              ray_direction_masked, decoder, c, logit_tau):
#         ''' Runs the bisection method for interval [d_low, d_high].

#         Args:
#             d_low (tensor): start values for the interval
#             d_high (tensor): end values for the interval
#             n_secant_steps (int): number of steps
#             ray0_masked (tensor): masked ray start points
#             ray_direction_masked (tensor): masked ray direction vectors
#             decoder (nn.Module): decoder model to evaluate point occupancies
#             c (tensor): latent conditioned code c
#             logit_tau (float): threshold value in logits
#         '''
#         d_pred = (d_low + d_high) / 2.
#         for i in range(n_secant_steps):
#             p_mid = ray0_masked + d_pred.unsqueeze(-1) * ray_direction_masked
#             with torch.no_grad():
#                 f_mid = decoder(p_mid, c, batchwise=False,
#                                 only_occupancy=True) - logit_tau
#             ind_low = f_mid < 0
#             d_low[ind_low] = d_pred[ind_low]
#             d_high[ind_low == 0] = d_pred[ind_low == 0]
#             d_pred = 0.5 * (d_low + d_high)
#         return d_pred

#     @staticmethod
#     def run_Secant_method(f_low, f_high, d_low, d_high, n_secant_steps,
#                           ray0_masked, ray_direction_masked, decoder, c,
#                           logit_tau):
#         ''' Runs the secant method for interval [d_low, d_high].

#         Args:
#             d_low (tensor): start values for the interval
#             d_high (tensor): end values for the interval
#             n_secant_steps (int): number of steps
#             ray0_masked (tensor): masked ray start points
#             ray_direction_masked (tensor): masked ray direction vectors
#             decoder (nn.Module): decoder model to evaluate point occupancies
#             c (tensor): latent conditioned code c
#             logit_tau (float): threshold value in logits
#         '''
#         d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
#         for i in range(n_secant_steps):
#             p_mid = ray0_masked + d_pred.unsqueeze(-1) * ray_direction_masked
#             with torch.no_grad():
#                 f_mid = decoder(p_mid, c, batchwise=False,
#                                 only_occupancy=True) - logit_tau
#             ind_low = f_mid < 0
#             if ind_low.sum() > 0:
#                 d_low[ind_low] = d_pred[ind_low]
#                 f_low[ind_low] = f_mid[ind_low]
#             if (ind_low == 0).sum() > 0:
#                 d_high[ind_low == 0] = d_pred[ind_low == 0]
#                 f_high[ind_low == 0] = f_mid[ind_low == 0]

#             d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
#         return d_pred

#     @staticmethod
#     def perform_ray_marching(ray0, ray_direction, decoder, c=None,
#                              tau=0.5, n_steps=[128, 129], n_secant_steps=8,
#                              depth_range=[0., 2.4], method='secant',
#                              check_cube_intersection=True, max_points=3500000):
#         ''' Performs ray marching to detect surface points.

#         The function returns the surface points as well as d_i of the formula
#             ray(d_i) = ray0 + d_i * ray_direction
#         which hit the surface points. In addition, masks are returned for
#         illegal values.

#         Args:
#             ray0 (tensor): ray start points of dimension B x N x 3
#             ray_direction (tensor):ray direction vectors of dim B x N x 3
#             decoder (nn.Module): decoder model to evaluate point occupancies
#             c (tensor): latent conditioned code
#             tay (float): threshold value
#             n_steps (tuple): interval from which the number of evaluation
#                 steps if sampled
#             n_secant_steps (int): number of secant refinement steps
#             depth_range (tuple): range of possible depth values (not relevant when
#                 using cube intersection)
#             method (string): refinement method (default: secant)
#             check_cube_intersection (bool): whether to intersect rays with
#                 unit cube for evaluation
#             max_points (int): max number of points loaded to GPU memory
#         '''
#         # Shotscuts
#         batch_size, n_pts, D = ray0.shape
#         device = ray0.device
#         logit_tau = get_logits_from_prob(tau)
#         n_steps = torch.randint(n_steps[0], n_steps[1], (1,)).item()

#         # Prepare d_proposal and p_proposal in form (b_size, n_pts, n_steps, 3)
#         # d_proposal are "proposal" depth values and p_proposal the
#         # corresponding "proposal" 3D points
#         d_proposal = torch.linspace(
#             depth_range[0], depth_range[1], steps=n_steps).view(
#                 1, 1, n_steps, 1).to(device)
#         d_proposal = d_proposal.repeat(batch_size, n_pts, 1, 1)

#         if check_cube_intersection:
#             d_proposal_cube, mask_inside_cube = \
#                 get_proposal_points_in_unit_cube(ray0, ray_direction,
#                                                  padding=0.1,
#                                                  eps=1e-6, n_steps=n_steps)
#             d_proposal[mask_inside_cube] = d_proposal_cube[mask_inside_cube]

#         p_proposal = ray0.unsqueeze(2).repeat(1, 1, n_steps, 1) + \
#             ray_direction.unsqueeze(2).repeat(1, 1, n_steps, 1) * d_proposal

#         # Evaluate all proposal points in parallel
#         with torch.no_grad():
#             val = torch.cat([(
#                 decoder(p_split, c, only_occupancy=True) - logit_tau)
#                 for p_split in torch.split(
#                     p_proposal.view(batch_size, -1, 3),
#                     int(max_points / batch_size), dim=1)], dim=1).view(
#                         batch_size, -1, n_steps)

#         # Create mask for valid points where the first point is not occupied
#         mask_0_not_occupied = val[:, :, 0] < 0

#         # Calculate if sign change occurred and concat 1 (no sign change) in
#         # last dimension
#         sign_matrix = torch.cat([torch.sign(val[:, :, :-1] * val[:, :, 1:]),
#                                  torch.ones(batch_size, n_pts, 1).to(device)],
#                                 dim=-1)
#         cost_matrix = sign_matrix * torch.arange(
#             n_steps, 0, -1).float().to(device)
#         # Get first sign change and mask for values where a.) a sign changed
#         # occurred and b.) no a neg to pos sign change occurred (meaning from
#         # inside surface to outside)
#         values, indices = torch.min(cost_matrix, -1)
#         mask_sign_change = values < 0
#         mask_neg_to_pos = val[torch.arange(batch_size).unsqueeze(-1),
#                               torch.arange(n_pts).unsqueeze(-0), indices] < 0

#         # Define mask where a valid depth value is found
#         mask = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied

#         # Get depth values and function values for the interval
#         # to which we want to apply the Secant method
#         n = batch_size * n_pts
#         d_low = d_proposal.view(
#             n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
#                 batch_size, n_pts)[mask]
#         f_low = val.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
#             batch_size, n_pts)[mask]
#         indices = torch.clamp(indices + 1, max=n_steps-1)
#         d_high = d_proposal.view(
#             n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
#                 batch_size, n_pts)[mask]
#         f_high = val.view(
#             n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
#                 batch_size, n_pts)[mask]

#         ray0_masked = ray0[mask]
#         ray_direction_masked = ray_direction[mask]

#         # write c in pointwise format
#         if c is not None and c.shape[-1] != 0:
#             c = c.unsqueeze(1).repeat(1, n_pts, 1)[mask]

#         # Apply surface depth refinement step (e.g. Secant method)
#         if method == 'secant' and mask.sum() > 0:
#             d_pred = DepthFunction.run_Secant_method(
#                 f_low, f_high, d_low, d_high, n_secant_steps, ray0_masked,
#                 ray_direction_masked, decoder, c, logit_tau)
#         elif method == 'bisection' and mask.sum() > 0:
#             d_pred = DepthFunction.run_Bisection_method(
#                 d_low, d_high, n_secant_steps, ray0_masked,
#                 ray_direction_masked, decoder, c, logit_tau)
#         else:
#             d_pred = torch.ones(ray_direction_masked.shape[0]).to(device)

#         # for sanity
#         pt_pred = torch.ones(batch_size, n_pts, 3).to(device)
#         pt_pred[mask] = ray0_masked + \
#             d_pred.unsqueeze(-1) * ray_direction_masked
#         # for sanity
#         d_pred_out = torch.ones(batch_size, n_pts).to(device)
#         d_pred_out[mask] = d_pred

#         return d_pred_out, pt_pred, mask, mask_0_not_occupied

#     @staticmethod
#     def forward(ctx, *input):
#         ''' Performs a forward pass of the Depth function.

#         Args:
#             input (list): input to forward function
#         '''
#         (ray0, ray_direction, decoder, c, n_steps, n_secant_steps, tau,
#          depth_range, method, check_cube_intersection, max_points) = input[:11]

#         # Get depth values
#         with torch.no_grad():
#             d_pred, p_pred, mask, mask_0_not_occupied = \
#                 DepthFunction.perform_ray_marching(
#                     ray0, ray_direction, decoder, c, tau, n_steps,
#                     n_secant_steps, depth_range, method, check_cube_intersection,
#                     max_points)

#         # Insert appropriate values for points where no depth is predicted
#         d_pred[mask == 0] = np.inf
#         d_pred[mask_0_not_occupied == 0] = 0

#         # Save values for backward pass
#         ctx.save_for_backward(ray0, ray_direction, d_pred, p_pred, c)
#         ctx.decoder = decoder
#         ctx.mask = mask

#         return d_pred

#     @staticmethod
#     def backward(ctx, grad_output):
#         ''' Performs the backward pass of the Depth function.

#         We use the analytic formula derived in the main publication for the
#         gradients. 

#         Note: As for every input a gradient has to be returned, we return
#         None for the elements which do no require gradients (e.g. decoder).

#         Args:
#             ctx (Pytorch Autograd Context): pytorch autograd context
#             grad_output (tensor): gradient outputs
#         '''
#         ray0, ray_direction, d_pred, p_pred, c = ctx.saved_tensors
#         decoder = ctx.decoder
#         mask = ctx.mask
#         eps = 1e-3

#         with torch.enable_grad():
#             p_pred.requires_grad = True
#             f_p = decoder(p_pred, c, only_occupancy=True)
#             f_p_sum = f_p.sum()
#             grad_p = torch.autograd.grad(f_p_sum, p_pred, retain_graph=True)[0]
#             grad_p_dot_v = (grad_p * ray_direction).sum(-1)

#             if mask.sum() > 0:
#                 grad_p_dot_v[mask == 0] = 1.
#                 # Sanity
#                 grad_p_dot_v[abs(grad_p_dot_v) < eps] = eps
#                 grad_outputs = -grad_output.squeeze(-1)
#                 grad_outputs = grad_outputs / grad_p_dot_v
#                 grad_outputs = grad_outputs * mask.float()

#             # Gradients for latent code c
#             if c is None or c.shape[-1] == 0 or mask.sum() == 0:
#                 gradc = None
#             else:
#                 gradc = torch.autograd.grad(f_p, c, retain_graph=True,
#                                             grad_outputs=grad_outputs)[0]

#             # Gradients for network parameters phi
#             if mask.sum() > 0:
#                 # Accumulates gradients weighted by grad_outputs variable
#                 grad_phi = torch.autograd.grad(
#                     f_p, [k for k in decoder.parameters()],
#                     grad_outputs=grad_outputs, retain_graph=True)
#             else:
#                 grad_phi = [None for i in decoder.parameters()]

#         # Return gradients for c, z, and network parameters and None
#         # for all other inputs
#         out = [None, None, None, gradc, None, None, None, None, None,
#                None, None] + list(grad_phi)
#         return tuple(out)

# def get_mask(tensor):
#     ''' Returns mask of non-illegal values for tensor.

#     Args:
#         tensor (tensor): Numpy or Pytorch tensor
#     '''
#     tensor, is_numpy = to_pytorch(tensor, True)
#     mask = ((abs(tensor) != np.inf) & (torch.isnan(tensor) == False))
#     mask = mask.bool()
#     if is_numpy:
#         mask = mask.numpy()

#     return mask

# def image_points_to_world(image_points, camera_mat, world_mat, scale_mat,
#                           invert=True):
#     ''' Transforms points on image plane to world coordinates.

#     In contrast to transform_to_world, no depth value is needed as points on
#     the image plane have a fixed depth of 1.

#     Args:
#         image_points (tensor): image points tensor of size B x N x 2
#         camera_mat (tensor): camera matrix
#         world_mat (tensor): world matrix
#         scale_mat (tensor): scale matrix
#         invert (bool): whether to invert matrices (default: true)
#     '''
#     batch_size, n_pts, dim = image_points.shape
#     assert(dim == 2)
#     device = image_points.device

#     d_image = torch.ones(batch_size, n_pts, 1).to(device)
#     return transform_to_world(image_points, d_image, camera_mat, world_mat,
#                               scale_mat, invert=invert)

# def transform_to_world(pixels, depth, camera_mat, world_mat, scale_mat,
#                        invert=True):
#     ''' Transforms pixel positions p with given depth value d to world coordinates.

#     Args:
#         pixels (tensor): pixel tensor of size B x N x 2
#         depth (tensor): depth tensor of size B x N x 1
#         camera_mat (tensor): camera matrix
#         world_mat (tensor): world matrix
#         scale_mat (tensor): scale matrix
#         invert (bool): whether to invert matrices (default: true)
#     '''
#     assert(pixels.shape[-1] == 2)

#     # Convert to pytorch
#     pixels, is_numpy = to_pytorch(pixels, True)
#     depth = to_pytorch(depth)
#     camera_mat = to_pytorch(camera_mat)
#     world_mat = to_pytorch(world_mat)
#     scale_mat = to_pytorch(scale_mat)

#     # Invert camera matrices
#     if invert:
#         camera_mat = torch.inverse(camera_mat)
#         world_mat = torch.inverse(world_mat)
#         scale_mat = torch.inverse(scale_mat)

#     # Transform pixels to homogen coordinates
#     pixels = pixels.permute(0, 2, 1)
#     pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)

#     # Project pixels into camera space
#     pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

#     # Transform pixels to world space
#     p_world = scale_mat @ world_mat @ camera_mat @ pixels

#     # Transform p_world back to 3D coordinates
#     p_world = p_world[:, :3].permute(0, 2, 1)

#     if is_numpy:
#         p_world = p_world.numpy()
#     return p_world

# def to_pytorch(tensor, return_type=False):
#     ''' Converts input tensor to pytorch.

#     Args:
#         tensor (tensor): Numpy or Pytorch tensor
#         return_type (bool): whether to return input type
#     '''
#     is_numpy = False
#     if type(tensor) == np.ndarray:
#         tensor = torch.from_numpy(tensor)
#         is_numpy = True
#     tensor = tensor.clone()
#     if return_type:
#         return tensor, is_numpy
#     return tensor

# def origin_to_world(n_points, camera_mat, world_mat, scale_mat, invert=True):
#     ''' Transforms origin (camera location) to world coordinates.

#     Args:
#         n_points (int): how often the transformed origin is repeated in the
#             form (batch_size, n_points, 3)
#         camera_mat (tensor): camera matrix
#         world_mat (tensor): world matrix
#         scale_mat (tensor): scale matrix
#         invert (bool): whether to invert the matrices (default: true)
#     '''
#     batch_size = camera_mat.shape[0]
#     device = camera_mat.device

#     # Create origin in homogen coordinates
#     p = torch.zeros(batch_size, 4, n_points).to(device)
#     p[:, -1] = 1.

#     # Invert matrices
#     if invert:
#         camera_mat = torch.inverse(camera_mat)
#         world_mat = torch.inverse(world_mat)
#         scale_mat = torch.inverse(scale_mat)

#     # Apply transformation
#     p_world = scale_mat @ world_mat @ camera_mat @ p

#     # Transform points back to 3D coordinates
#     p_world = p_world[:, :3].permute(0, 2, 1)
#     return p_world

# def normalize_tensor(tensor, min_norm=1e-5, feat_dim=-1):
#     ''' Normalizes the tensor.

#     Args:
#         tensor (tensor): tensor
#         min_norm (float): minimum norm for numerical stability
#         feat_dim (int): feature dimension in tensor (default: -1)
#     '''
#     norm_tensor = torch.clamp(torch.norm(tensor, dim=feat_dim, keepdim=True),
#                               min=min_norm)
#     normed_tensor = tensor / norm_tensor
#     return normed_tensor
