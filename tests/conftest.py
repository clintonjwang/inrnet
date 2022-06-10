import pytest
import torch

from inrnet.inn import point_set
nn=torch.nn
from inrnet import inn, args as args_module
from inrnet.inn.nets.classifier import InrCls

requirescuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires cuda'
)

@pytest.fixture
def qmc_2d_sequence256():
    return point_set.generate_quasirandom_sequence(d=2, n=256, bbox=(-1,1,-1,1), dtype=torch.float, device="cpu")

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
            'learning rate': '1e-5',
            'weight decay': '1e-3',
        },
    }
    args_module.infer_missing_args(args)
    return args

@pytest.fixture
def point_set2d():
    # points at (.5,.5) and (-.5,-.5)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    x = torch.tensor(((.5,.5), (-.5,-.5)), device=device)
    return x.as_subclass(point_set.PointSet)


@pytest.fixture
def inr2(C=3):
    inr_values = torch.zeros(2, C)
    inr_values[0,0] = 1
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    class dummy_inr(nn.Module):
        def forward(self, coords):
            return inr_values.to(dtype=coords.dtype, device=device)
    return inn.BlackBoxINR([dummy_inr()], channels=C, input_dims=2, domain=(-1,1), device=device)

@pytest.fixture
def inr256(C=1):
    inr_values = torch.zeros(256, C)
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
    return inn.BlackBoxINR([dummy_inr()], channels=C, input_dims=2, domain=(-1,1), device=device)

@pytest.fixture
def inr_classifier(in_ch=1, n_classes=4):
    if torch.cuda.is_available():
        device = 'cuda'
        return InrCls(in_ch, n_classes, device=device)
    else:
        device = 'cpu'
        return None

