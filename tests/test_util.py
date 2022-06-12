import pytest, torch
nn = torch.nn
from inrnet import args as args_module, jobs, util

# def test_parser():
#     args_module.parse_args(['-c','inet_i2', '-j','pytest', '-w'])

# def test_meshgrid():
#     assert util.meshgrid(torch.arange(2), torch.arange(2))