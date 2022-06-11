import pytest, torch
nn = torch.nn
from inrnet import util

def test_increment():
    assert util.increment_name('test.10') == 'test.11'
    assert util.increment_name('G.2') == 'G.3'

# def test_meshgrid():
#     assert util.meshgrid(torch.arange(2), torch.arange(2))