import os, yaml, torch, argparse, shutil
osp = os.path
import dill as pickle
import numpy as np

from inrnet import util, losses
from inrnet.data import dataloader

ANALYSIS_DIR = osp.expanduser("~/code/diffcoord/temp")
RESULTS_DIR = osp.expanduser("~/code/diffcoord/results")

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

def get_job_model_and_args(job, metric="dice"):
    args = get_job_args(job)
    raise NotImplementedError
    model = model_module.get_model(args["paths"], args["network"]).cuda().eval()
    if args["network"]["type"] == "TopNet":
        model_path = osp.join(RESULTS_DIR, job, "weights/best.pth")
    else:
        model_path = osp.join(RESULTS_DIR, job, f"weights/best_{metric}.pth")
    model.load_state_dict(torch.load(model_path), strict=False)
    return model, args

def get_dataset_for_job(job):
    return get_job_args(job)["dataset"]

def get_attributes_for_job(job):
    return get_job_args(job)["data loading"]["attributes"]


def get_metrics_for_dataloader(models, dataloader, args, out_dir=None):
    for m in models:
        m.eval()
    network_settings = args["network"]
    pred_vols = []
    gt_vols = []
    dice_tracker = util.MetricTracker("test DICE")
    hausdorff_tracker = util.MetricTracker("test HD")
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    metric_dict = {}
    with torch.no_grad():
        for batch in dataloader:
            imgs, gt_segs = batch["image"].cuda(), batch["label"]
            imgs = imgs.detach().squeeze(1).cpu().numpy()
            for i in range(pred_logits.shape[0]):
                seriesID = batch["seriesID"][i]

                pred_seg = (pred_logits[i] > 0).detach().squeeze().cpu().numpy()
                gt_seg = gt_segs[i].squeeze().numpy()

                dice = losses.dice_np(pred_seg, gt_seg)
                dice_tracker.update_with_minibatch(dice)
                hausd = skimage.metrics.hausdorff_distance(pred_seg, gt_seg)
                hausdorff_tracker.update_with_minibatch(hausd)

                metric_dict[seriesID] = (dice, hausd)

                if out_dir is not None:
                    root = osp.join(out_dir, seriesID)
                    util.save_example_slices(imgs[i], gt_seg, pred_seg, root=root)
                    with open(osp.join(root, "metrics.txt"), "w") as f:
                        f.write("%.2f\n%d" % (dice, hausd))


    if out_dir is not None:
        metrics_iter = sorted(metric_dict.items(), key=lambda item: item[1][0])
        with open(osp.join(out_dir, "metrics.txt"), "w") as f:
            f.write("Avg:\t%.2f\t%d\n" % (dice_tracker.epoch_average(), hausdorff_tracker.epoch_average()))
            for seriesID, metrics in metrics_iter:
                f.write("%s:\t%.2f\t%d\n" % (seriesID, metrics[0], metrics[1]))

    return dice_tracker, hausdorff_tracker

