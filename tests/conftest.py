import pytest
from inrnet import args as args_module

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
