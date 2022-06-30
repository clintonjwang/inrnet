import pytest, torch

from inrnet.utils import util
from inrnet.utils import jobs
nn = torch.nn
from inrnet.utils import args as args_module

# def test_parser():
#     args_module.parse_args(['-c','inet_i2', '-j','pytest', '-w'])

# def test_meshgrid():
#     assert util.meshgrid(torch.arange(2), torch.arange(2))