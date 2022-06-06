from typing import Optional, List
import wandb

class Experiment:
    def __init__(self, name:str, sweeps:Optional[List]=None):
        self.name = name
        self.sweeps = sweeps

def start_experiment(name:str):
    return


def sweep_sample_type():
    sweep_config = {
        "name": "sweep sample_type",
        "method": "grid",
        "parameters": {
            "sample_type": {
                "values": ["rqmc", 'qmc', 'grid', "shrunk"],
            },
        }
    }
    return wandb.sweep(sweep_config, project='inrnet')

def sweep_parameter():
    name = 'learning_rate_sweep'
    sweep_config = {
        "name": name,
        "method": "grid",
        "parameters": {
            "learning_rate": {
                "values": [1e-5, 1e-4, 1e-3, 1e-2],
            },
        }
    }
    sweep_id = wandb.sweep(sweep_config, project='inrnet')
    experiment = Experiment(name, sweeps=[sweep_id])
    return experiment
