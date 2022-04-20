import numpy as np 
import torch
nn = torch.nn
F = nn.functional

import util
from inrnet.inn import functional as inrF

def L1_dist_inr(N=128):
    def l1_qmc(pred,target):
        coords = target.generate_sample_points(sample_size=N)
        return (pred(coords)-target(coords)).abs().mean()
    return l1_qmc
# class L1_dist_inr(nn.Module):
#     def __init__(self, N=128):
#         self.N = N
#     def forward(pred,target):
#         coords = target.generate_sample_points(sample_size=N)
#         return (pred(coords)-target(coords)).abs().mean()

def l2_dist_inr(N=128):
    def l2_qmc(pred,target):
        coords = target.generate_sample_points(sample_size=N)
        return (pred(coords)-target(coords)).pow(2).mean()
    return l2_qmc

def L1_dist(inr, gt_values, coords):
    pred = inr(coords)
    pred = util.realign_values(pred, coords_gt=coords, inr=inr)
    return (pred-gt_values).abs().mean()

def mse_loss(pred,target):
    return (pred-target).pow(2).flatten(start_dim=1).mean(1)

def adv_loss_fxns(loss_settings):
    if "WGAN" in loss_settings["adversarial loss type"]:
        G_fxn = lambda fake_logit: -fake_logit.squeeze()
        D_fxn = lambda fake_logit, true_logit: (fake_logit - true_logit).squeeze()
        return G_fxn, D_fxn
    elif "standard" in loss_settings["adversarial loss type"]:
        G_fxn = lambda fake_logit: -fake_logit - torch.log1p(torch.exp(-fake_logit))#torch.log(1-torch.sigmoid(fake_logit)).squeeze()
        D_fxn = lambda fake_logit, true_logit: (fake_logit + torch.log1p(torch.exp(-fake_logit)) + torch.log1p(torch.exp(-true_logit))).squeeze()
        #-torch.log(1-fake_logit) - torch.log(true_logit)
        return G_fxn, D_fxn
    else:
        raise NotImplementedError

def gradient_penalty(real_img, generated_img, D=None, DR=None):
    B = real_img.size()[0]
    alpha = torch.rand(B, 1, 1, 1).expand_as(real_img).cuda()
    interp_img = nn.Parameter(alpha*real_img + (1-alpha)*generated_img.detach()).cuda()
    if DR is None:
        interp_logit = D(interp_img)
    else:
        interp_logit,_ = DR(interp_img)

    grads = torch.autograd.grad(outputs=interp_logit, inputs=interp_img,
               grad_outputs=torch.ones(interp_logit.size()).cuda(),
               create_graph=True, retain_graph=True)[0].view(B, -1)
    grads_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-12)
    return (grads_norm - 1) ** 2
