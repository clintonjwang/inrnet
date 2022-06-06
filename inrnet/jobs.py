import os
import shutil
import wandb, yaml
osp = os.path

from inrnet import ANALYSIS_DIR, RESULTS_DIR, util

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
    sweep_config = {
        "name": "lr sweep",
        "method": "grid",
        "parameters": {
            "learning_rate": {
                "values": [1e-5, 1e-4, 1e-3, 1e-2],
            },
        }
    }
    return wandb.sweep(sweep_config, project='inrnet')


def rename_job(job, new_name):
    os.rename(osp.join(RESULTS_DIR, job), osp.join(RESULTS_DIR, new_name))
    for folder in util.glob2(ANALYSIS_DIR, "*", job):
        os.rename(folder, folder.replace(job, new_name))

def delete_job(job):
    shutil.rmtree(osp.join(RESULTS_DIR, job))
    for folder in util.glob2(ANALYSIS_DIR, "*", job):
        shutil.rmtree(folder)

def get_job_args(job):
    config_path = osp.join(RESULTS_DIR, job, f"config.yaml")
    args = yaml.safe_load(open(config_path, "r"))
    return args

def get_dataset_for_job(job):
    return get_job_args(job)["data loading"]["dataset"]

