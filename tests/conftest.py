import pytest
import torch
nn=torch.nn
from inrnet import inn, args as args_module
from inrnet.inn import point_set, qmc

@pytest.fixture
def qmc_2d_sequence256():
    return qmc.generate_quasirandom_sequence(d=2, n=256, bbox=(-1,1,-1,1), dtype=torch.float, device="cpu")

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
def inr16x16(C=1, dims=(16,16)):
    inr_values = torch.zeros(dims[0]*dims[1], C)
    inr_values[0,:] = 1
    inr_values[-4,:] = 1
    inr_values[-2,:] = 2
    inr_values[2,:] = 2
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    class dummy_inr(nn.Module):
        def forward(self, coords):
            return inr_values.to(dtype=coords.dtype, device=device)
    return inn.BlackBoxINR([dummy_inr()], channels=C, input_dims=len(dims), domain=(-1,1), device=device)

