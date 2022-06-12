import pytest, torch
from inrnet import inn
from inrnet.inn.nets.inrnet import INRNet

# def test_inrnet():
#     inet = INRNet(sampler={'grid'}, sample_size=512)
#     assert inet.sample_mode == 'grid'
#     assert inet.sample_size == 512