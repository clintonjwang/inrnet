import os, yaml, torch, argparse, shutil, pickle
osp = os.path
import numpy as np

from inrnet import RESULTS_DIR, ANALYSIS_DIR, util, losses
from inrnet.data import dataloader

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

