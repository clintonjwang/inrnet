import numpy as np 
import torch, pdb
nn = torch.nn
F = nn.functional

from inrnet import util, inn
from inrnet.inn import functional as inrF

def CrossEntropy(N=128):
    ce = nn.CrossEntropyLoss()
    def ce_loss(pred, class_ix):
        coords = pred.generate_sample_points(sample_size=N)
        return ce(pred(coords), class_ix)
    return ce_loss

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
    #pred = util.realign_values(pred, coords_gt=coords, inr=inr)
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

def gradient_penalty(real_img, generated_img, D):
    B = real_img.size(0)
    alpha = torch.rand(B, 1, 1, 1, device='cuda')
    interp_img = nn.Parameter(alpha*real_img + (1-alpha)*generated_img.detach())
    interp_logit = D(interp_img)

    grads = torch.autograd.grad(outputs=interp_logit, inputs=interp_img,
               grad_outputs=torch.ones(interp_logit.size(), device='cuda'),
               create_graph=True, retain_graph=True)[0].view(B, -1)
    grads_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-12)
    return (grads_norm - 1) ** 2

def gradient_penalty_inr(coords, real_inr, generated_inr, D):
    real_img = real_inr.cached_outputs
    generated_img = generated_inr.cached_outputs
    B = real_img.size(0)
    alpha = torch.rand(B, 1, 1, device='cuda')
    class dummy_inr(nn.Module):
        def forward(self, coords):
            return alpha*real_img + (1-alpha)*generated_img.detach()
    interp_img = inn.BlackBoxINR([dummy_inr()], channels=1, input_dims=2).cuda()
    interp_img(coords)
    interp_vals = interp_img.cached_outputs
    interp_vals.requires_grad = True
    interp_logit = D(interp_img)(coords)

    grads = torch.autograd.grad(outputs=interp_logit, inputs=interp_vals,
               grad_outputs=torch.ones(interp_logit.size(), device='cuda'),
               create_graph=True, retain_graph=True)[0].view(B, -1)
    grads_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-12)
    return (grads_norm - 1) ** 2
