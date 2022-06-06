import pytest, torch
from inrnet.inn import functional as inrF
from inrnet.inn.layers.other import PositionalEncoding

@pytest.fixture
def pos_enc_layer():
    return PositionalEncoding(N=2, additive=True)

def test_pos_enc(inr16x16, pos_enc_layer, qmc_2d_sequence256):
    inr16x16.sampled_coords = qmc_2d_sequence256
    values = torch.randn(1,16*16,8)
    embeddings = inrF.pos_enc(values, inr16x16, pos_enc_layer)
    assert embeddings.shape == (1,16*16,8)

def test_conv():
    #inrF.conv
    pass