
import os
osp=os.path
import numpy as np
import torch, kornia
nn=torch.nn
F=nn.functional
from monai.networks.blocks import Warp, DVF2DDF

root = osp.expanduser("~/code/placenta/temp/synth")

from monai.data import NibabelReader

from inrnet.util import glob2

def generate_sequence():
    paths = glob2('/data/vision/polina/scratch/clintonw/datasets/oasis/OASIS_*/slice_norm.nii.gz')
    reader = NibabelReader()

    for path in paths:
        img = reader.get_data(reader.read(path))[0].squeeze().view(1,1,*img.shape)
        def_field = reader
        # new_path = 

    return

def generate_synthetic_dataset(img, N=3):
    width = 32
    vf2df = DVF2DDF()
    warp = Warp()
    VFs = F.interpolate(torch.randn(N,1,width//8,width//8), scale_factor=8) * 2 + \
          F.interpolate(torch.randn(N,1,width//4,width//4), scale_factor=4) + \
          F.interpolate(torch.randn(N,1,width//2,width//2), scale_factor=2) * .5
    w1 = warp(img, vf2df(VFs[:1]))
    w2 = warp(w1, vf2df(VFs[1:2]))
    w3 = warp(w2, vf2df(VFs[2:3]))

    inr = fit_siren(img)
    w1 = fit_siren(w1)
    w2 = fit_siren(w2)
    w3 = fit_siren(w3)
