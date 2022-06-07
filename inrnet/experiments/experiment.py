from typing import Optional, List
import yaml, wandb, pdb

from inrnet import CONFIG_DIR

class Experiment:
    def __init__(self, name:str, sweeps:Optional[List]=None):
        self.name = name
        self.sweeps = sweeps

def start_experiment(name:str):
    experiment = Experiment(name)
    return experiment

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

def sweep_parameter(parameter_name, values):
    name = f'{parameter_name}_sweep'
    sweep_config = {
        "name": name,
        "method": "grid",
        "parameters": {
            parameter_name: {
                "values": values,
            },
        }
    }
    # name = 'learning_rate_sweep'
    # sweep_config = {
    #     "name": name,
    #     "method": "grid",
    #     "parameters": {
    #         "learning_rate": {
    #             "values": [1e-5, 1e-4, 1e-3, 1e-2],
    #         },
    #     }
    # }
    sweep_id = wandb.sweep(sweep_config, project='inrnet')
    experiment = Experiment(name, sweeps=[sweep_id])
    return experiment

def get_param_dict(vmin, vmax, distribution='log_uniform_values'):
    return {
        'distribution': distribution,
        'min': vmin,
        'max': vmax,
    }

def bayes_parameters(name, parameter_dict=None):
    if parameter_dict is None:
        parameter_dict = yaml.safe_load(open(f'{CONFIG_DIR}/{name}.yaml', 'r'))
    for p in parameter_dict:
        if "distribution" not in parameter_dict[p]:
            parameter_dict[p] = get_param_dict(parameter_dict[p]['min'], parameter_dict[p]['max'])

    sweep_config = {
        "name": name,
        "method": "bayes",
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize',
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 100,
        },
        "parameters": parameter_dict,
    }
    sweep_id = wandb.sweep(sweep_config, project='inrnet')
    experiment = Experiment(name, sweeps=[sweep_id])
    return  

