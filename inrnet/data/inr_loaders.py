import os, torch
osp = os.path
import numpy as np
import dill as pickle
from PIL import Image
from torchvision import transforms

from inrnet.models.inrs import siren
from inrnet import util

TMP_DIR = osp.expanduser("~/code/diffcoord/temp")
DS_DIR = "/data/vision/polina/scratch/clintonw/datasets"

def get_h2z_inr_dataloader(animal):
    inr = siren.Siren(out_channels=3)
    paths = sorted(util.glob2(f"{DS_DIR}/inrnet/{animal}/siren_*.pt"))
    keys = ['net.0.linear.weight', 'net.0.linear.bias', 'net.1.linear.weight', 'net.1.linear.bias', 'net.2.linear.weight', 'net.2.linear.bias', 'net.3.linear.weight', 'net.3.linear.bias', 'net.4.weight', 'net.4.bias']
    for path in paths:
        data = torch.load(path)
        for ix in range(len(data)):
            param_dict = {k:data[k][ix] for k in keys}
            try:
                inr.load_state_dict(param_dict)
            except RuntimeError:
                continue
            yield inr.cuda()