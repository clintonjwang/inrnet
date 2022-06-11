import pytest, torch
from inrnet import inn
from inrnet.inn.nets.inrnet import INRNet

def test_inrnet():
    inet = INRNet(domain=(-1,1), sample_mode='grid', sample_size=512)
    assert inet.sample_mode == 'grid'
    assert inet.sample_size == 512