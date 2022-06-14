import pytest, torch
from inrnet import inn
from inrnet.inn import functional as inrF

from conftest import requirescuda

@pytest.fixture
def pos_enc_layer():
    return inn.PositionalEncoding(N=2, additive=True)

# def test_pos_enc(inr256, pos_enc_layer, qmc_2d_sequence256):
#     inr256.sampled_coords = qmc_2d_sequence256
#     values = torch.randn(1,256,8)
#     embeddings = inrF.pos_enc(values, inr256, pos_enc_layer)
#     assert embeddings.shape == (1,256,8)


@pytest.fixture
def conv_layer1to1():
    if torch.cuda.is_available():
        return inn.MLPConv(1,1, kernel_size=(.1,.1))
    else:
        return None

@requirescuda
def test_conv(inr256, qmc_2d_sequence256):
    inr256.sampled_coords = qmc_2d_sequence256
    values = torch.randn(1,256,1)
    with torch.no_grad():
        raise NotImplementedError
        out = inrF.conv(values, inr256)
    assert out.shape == (1,256,1)