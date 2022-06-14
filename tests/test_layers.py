import pytest, torch
nn = torch.nn
from inrnet.inn.layers.reshape import GlobalAvgPoolSequence
from conftest import requirescuda

@requirescuda
def test_inr_parent(inr256, qmc_2d_sequence256):
    layer = GlobalAvgPoolSequence(nn.ReLU())
    vvf = layer(inr256)
    out = vvf(qmc_2d_sequence256)
    assert out.shape == (1,1,1)
    assert torch.all(out>0)
