import numpy as np 
import torch
nn = torch.nn
F = nn.functional

def L1_dist(pred,target):
    return (pred-target).abs().flatten(start_dim=1).mean(1)

def mse_loss(pred,target):
    return (pred-target).pow(2).flatten(start_dim=1).mean(1)
