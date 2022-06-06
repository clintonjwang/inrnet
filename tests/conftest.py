import pytest
import torch
from inrnet import args as args_module
from inrnet.inn import point_set, qmc

@pytest.fixture
def qmc_2d_sequence():
    return qmc.generate_quasirandom_sequence(d=2, n=128, bbox=(-1,1,-1,1), dtype=torch.float, device="cpu")

@pytest.fixture
def args():
    args = {
        'job_id': 'manual',
        'data loading': {},
        'paths': {
            'slurm output dir': '~/code/inrnet/results',
        },
        'optimizer': {
            'type': 'AdamW',
            'beta1': .5,
            'learning_rate': '1e-5',
            'weight decay': '1e-3',
        },
    }
    args_module.infer_missing_args(args)
    return args



@pytest.fixture
def inr(qmc_2d_sequence, C=1, dims=(16,16)):
    zz = torch.zeros(dims[0]*dims[1], C)
    zz[0,:] = 1
    zz[-4,:] = 1
    zz[-2,:] = 2
    zz[2,:] = 2
    class dummy_inr(nn.Module):
        def forward(self, coords):
            return zz.to(dtype=coords.dtype, device='cpu')
    return inn.BlackBoxINR([dummy_inr()], channels=C, input_dims=len(dims), domain=(-1,1))

