import os, pytest
import torch

from inrnet.utils import util
from inrnet.utils import jobs as job_mgmt
osp = os.path
nn = torch.nn
F = nn.functional
import torchvision.models

from inrnet import inn
from inrnet.data import dataloader
from inrnet.inn import point_set

def test_equivalence_dummy(inr256):
    h,w=(16,16)
    x = inr256.produce_images(h,w)
    assert x.shape == (1,1,h,w)
    #     conv = nn.Conv2d(1,2,3,1,padding=1,bias=False)
    #     norm = nn.BatchNorm2d(2)
    #     discrete_model = nn.Sequential(conv, norm, nn.LeakyReLU(inplace=True)).eval()
    #     InrNet, output_shape = inn.conversion.translate_discrete_model(discrete_model, (h,w))
    #     y = discrete_model(x)

    #     coords = util.meshgrid_coords(h,w)
    #     out_inr = InrNet.cuda()(inr)
    #     out_inr.toggle_grid_mode(True)
    #     out = out_inr.eval()(coords)
        
    # assert output_shape is not None
    # out = util.realign_values(out, inr=out_inr)
    # out = out.reshape(1,*output_shape,-1).permute(0,3,1,2)
        
    # assert y.shape == out.shape
    # assert torch.allclose(y, out, rtol=1e-5, atol=1e-3)
