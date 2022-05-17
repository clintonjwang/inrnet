
import os, pdb
osp=os.path
import numpy as np
import torch, kornia
nn=torch.nn
F=nn.functional
from monai.networks.blocks import Warp, DVF2DDF
from monai.data import NibabelReader

from inrnet.util import glob2
from inrnet.fit_inr import fit_siren_to_img

DS_DIR = "/data/vision/polina/scratch/clintonw/datasets"

def generate_synthetic_dataset(N_train=350, N_val=50):
    N = N_train + N_val
    paths = sorted(glob2('/data/vision/polina/scratch/clintonw/datasets/oasis/OASIS_*/slice_norm.nii.gz'))[:N]
    reader = NibabelReader()

    for index, path in enumerate(paths):
        seg_path = osp.dirname(path)+'/slice_seg4.nii.gz'
        img = torch.tensor(reader.get_data(reader.read(path))[0]).squeeze().unsqueeze(0).unsqueeze(0).contiguous()
        seg = torch.tensor(reader.get_data(reader.read(seg_path))[0]).squeeze().unsqueeze(0).unsqueeze(0).contiguous()
        subset = 'train' if index<N_train else 'val'
        fit_synthetic_series_for_img(img, seg, index, subset=subset)


def fit_synthetic_series_for_img(orig_img, seg, index, total_steps=1500, subset='train'):
    #orig_img - (1,1,h,w)
    H,W = orig_img.shape[-2:]
    orig_img /= orig_img.max()
    orig_seg = (seg==3).float()
    vf2df = DVF2DDF()
    warp = Warp()
    VFs = F.interpolate(torch.randn(3,2,H//16,W//16), scale_factor=16)*2 + \
          F.interpolate(torch.randn(3,2,H//8,W//8), scale_factor=8) + \
          F.interpolate(torch.randn(3,2,H//4,W//4), scale_factor=4)/2
    VFs[1:2] = VFs[:1]+VFs[1:2]
    VFs[2:3] = VFs[1:2]+VFs[2:3]
    w1 = warp(orig_img, vf2df(VFs[:1]))
    w2 = warp(orig_img, vf2df(VFs[1:2]))
    w3 = warp(orig_img, vf2df(VFs[2:3]))
    s1 = warp(orig_seg, vf2df(VFs[:1]))
    s2 = warp(orig_seg, vf2df(VFs[1:2]))
    s3 = warp(orig_seg, vf2df(VFs[2:3]))
    segs = (orig_seg.bool(), s1>.5, s2>.5, s3>.5)

    for ix, img in enumerate((orig_img, w1, w2, w3)):
        inr,loss = fit_siren_to_img(img, total_steps)
        path = DS_DIR+f"/oasis/{subset}_{index}_{ix}.pt"
        torch.save((inr.state_dict(), segs[ix]), path) #white matter
        loss_path = DS_DIR+f"/oasis/loss_{subset}_{index}_{ix}.txt"
        open(loss_path, 'w').write(str(loss.item()))

