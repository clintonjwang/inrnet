import os, pdb, torch
osp = os.path
import torch
nn = torch.nn
F = nn.functional

from data import dataloader
import inn

TMP_DIR = osp.expanduser("~/code/diffcoord/temp")
rescale_float = mtr.ScaleIntensity()

def train_seg_model(args):
    paths=args["paths"]