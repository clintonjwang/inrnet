import pytest, torch
nn = torch.nn
from inrnet import args as args_module, jobs, util

# def test_parser():
#     args_module.parse_args(['-c','inet_i2', '-j','pytest', '-w'])

def test_get_job_args():
    args = jobs.get_job_args('manual')
    assert 'data loading' in args
    ds = jobs.get_dataset_for_job('manual')
    assert ds == args["data loading"]["dataset"]
    jobs.rename_job('manual', 'testmanual')
    assert jobs.get_job_args('testmanual') == args
    jobs.rename_job('testmanual', 'manual')

# def test_meshgrid():
#     assert util.meshgrid(torch.arange(2), torch.arange(2))