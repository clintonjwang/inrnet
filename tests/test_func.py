import pytest, torch
from inrnet import inn
from inrnet.inn import functional as inrF

@pytest.fixture
def pos_enc_layer():
    return inn.PositionalEncoding(N=2, additive=True)

@pytest.fixture
def conv_layer():
    return inn.MLPConv(1,1, kernel_size=(.1,.1))

def test_pos_enc(inr16x16, pos_enc_layer, qmc_2d_sequence256):
    inr16x16.sampled_coords = qmc_2d_sequence256
    values = torch.randn(1,256,8)
    embeddings = inrF.pos_enc(values, inr16x16, pos_enc_layer)
    assert embeddings.shape == (1,256,8)

def test_conv(inr16x16, conv_layer, qmc_2d_sequence256):
    inr16x16.sampled_coords = qmc_2d_sequence256
    values = torch.randn(1,256,1)
    with torch.no_grad():
        out = inrF.conv(values, inr16x16, conv_layer)
    assert out.shape == (1,256,1)