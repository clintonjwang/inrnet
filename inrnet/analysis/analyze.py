import os, submitit, functools, itertools, shutil, torch
osp=os.path
import numpy as np
import pandas as pd
import dill as pickle

import warnings
warnings.simplefilter("ignore", UserWarning)
import skimage.measure, skimage.metrics


TMP_DIR = osp.expanduser("~/code/placenta/temp")
RESULTS_DIR = osp.expanduser("~/code/placenta/results")

def submit_job(fxn, *args, job_name="unnamed", **kwargs):
    executor = submitit.AutoExecutor(folder=osp.join(TMP_DIR, "slurm", job_name))
    executor.update_parameters(
        name=job_name,
        slurm_partition="gpu",
        tasks_per_node=1,
        gpus_per_node=1,
        cpus_per_task=16,
        slurm_exclude="bergamot,perilla,caraway,cassia,anise",
        slurm_exclusive=True,
        timeout_min=90000,
    )
    job = executor.submit(fxn, *args, **kwargs)
    return job

def submit_array_job(fxn, *iterated_args, job_name="unnamed", **kwargs):
    kw_fxn = functools.partial(fxn, **kwargs)
    executor = submitit.AutoExecutor(folder=osp.join(TMP_DIR, "slurm", job_name))
    executor.update_parameters(
        name=job_name,
        slurm_partition="gpu",
        tasks_per_node=1,
        gpus_per_node=1,
        cpus_per_task=16,
        slurm_exclude="bergamot,perilla,caraway,cassia,anise",
        slurm_exclusive=True,
        timeout_min=90000,
    )
    jobs = executor.map_array(kw_fxn, *iterated_args)
    return jobs

def get_connected_comp_vols(model, dataloader, args):
    pred_vols = []
    gt_vols = []
    with torch.no_grad():
        for batch in dataloader:
            imgs, gt_segs = batch["image"].cuda(), batch["label"].cuda()
            pred_logits = model_module.inference(model, imgs, args=args)
            pred_segs = (pred_logits > 0).squeeze(1).detach().cpu().numpy()
            imgs = imgs.squeeze(1).detach().cpu().numpy()
            gt_segs = gt_segs.squeeze(1).detach().cpu().numpy()
            for ix in range(len(imgs)):
                label_img = skimage.measure.label(pred_segs[ix])
                props = skimage.measure.regionprops(label_img)#, intensity_image=imgs[ix])
                pred_vols.append(sorted([p.area for p in props], reverse=True))
                
                label_img = skimage.measure.label(gt_segs[ix])
                props = skimage.measure.regionprops(label_img)#, intensity_image=imgs[ix])
                gt_vols.append(sorted([p.area for p in props], reverse=True))

    return pred_vols, gt_vols


def get_sample_batch(job, phase="train", args=None):
    if args is None:
        args = job_mgmt.get_job_args(job)
    dataloaders = get_dataloaders(args)
    for batch in dataloaders[phase]:
        return batch

def get_sample_outputs(job, phase="train"):
    model, args = job_mgmt.get_job_model_and_args(job)
    dataloaders = get_dataloaders(args)
    with torch.no_grad():
        for batch in dataloaders[phase]:
            imgs, gt_segs = batch["image"].cuda(), batch["label"].cuda()
            with torch.cuda.amp.autocast():
                pred_logits = model(imgs)
            return imgs.cpu(), gt_segs.cpu(), pred_logits.cpu()

def get_metrics_for_dataloader(model, dataloader, args, out_dir=None):
    model.eval()
    network_settings = args["network"]
    pred_vols = []
    gt_vols = []
    dice_tracker = util.MetricTracker("DICE", function=losses.dice_np)
    hausdorff_tracker = util.MetricTracker("HD", function=skimage.metrics.hausdorff_distance)
    components_tracker = util.MetricTracker(name="extra components", function=losses.components_diff)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    metric_dict = {}
    with torch.no_grad():
        for batch in dataloader:
            imgs, gt_segs = batch["image"].cuda(), batch["label"]
            pred_logits = model_module.inference(model, imgs, args=args)
            imgs = imgs.detach().squeeze(1).cpu().numpy()
            for i in range(pred_logits.shape[0]):
                seriesID = batch["seriesID"][i]

                pred_seg = (pred_logits[i] > 0).detach().squeeze().cpu().numpy()
                gt_seg = gt_segs[i].squeeze().numpy()

                dice = dice_tracker(pred_seg, gt_seg)
                hausd = hausdorff_tracker(pred_seg, gt_seg)
                comp = components_tracker(pred_seg, gt_seg)
                metric_dict[seriesID] = (dice, hausd, comp)

                if out_dir is not None:
                    root = osp.join(out_dir, seriesID)
                    util.save_example_slices(imgs[i], gt_seg, pred_seg, root=root)

    if out_dir is not None:
        metrics_iter = sorted(metric_dict.items(), key=lambda item: item[1][0]) #sort by dice
        with open(osp.join(out_dir, "metrics.txt"), "w") as f:
            f.write("Avg:\t%.2f\t%d\n" % (dice_tracker.epoch_average(), hausdorff_tracker.epoch_average()))
            for seriesID, metrics in metrics_iter:
                f.write("%s:\t%.2f\t%d\t%d\n" % (seriesID, *metrics))

    return dice_tracker.epoch_average(), hausdorff_tracker.epoch_average()

def generate_test_predictions_for_job(job):
    model, args = job_mgmt.get_job_model_and_args(job)
    dataloaders = get_dataloaders(args)
    dice, hausd = get_metrics_for_dataloader(model, dataloaders["test"],
        args, out_dir=args["paths"]["job output dir"]+"/imgs/test")
    return dice, hausd

def delete_jobs_without_results():
    results_folders = util.glob2(RESULTS_DIR)
    for folder in results_folders:
        if not osp.exists(folder+"/weights/best_G.pth"):
            job = osp.basename(folder)
            print(job)
            shutil.rmtree(folder)

def collect_all_missing_job_metrics():
    results_folders = util.glob2(RESULTS_DIR)
    df = tables.get_results_table()
    for folder in results_folders:
        job = osp.basename(folder)
        if job == "manual":
            continue
        elif (job not in df.index or np.isnan(df.loc[job, 'val dice'])) and osp.exists(folder+"/weights/best_dice.pth"):
            collect_metrics_for_jobs(job, slurm=True)
            print(job)

def get_regressor_metrics_for_job(jobs, slurm=False, overwrite=False):
    if isinstance(jobs, str):
        jobs = [jobs]
    if slurm is True:
        return submit_job(get_regressor_metrics_for_job, jobs,
            overwrite=overwrite, slurm=False, job_name="incv3_err")
    elif slurm == "array":
        return submit_array_job(get_regressor_metrics_for_job, jobs,
            overwrite=overwrite, slurm=False, job_name="incv3_err")

    for job in jobs:
        # outputs = job_mgmt.get_attributes_for_job(job)
        # residuals = inception.get_inception_v3_residuals(job, overwrite=overwrite)
        df = tables.get_results_table()
        for var in outputs:
            pass
            # df.loc[job, f"{var} mean error"] = np.nanmean(residuals[var])
            # df.loc[job, f"{var} error STD"] = np.nanstd(residuals[var])
        tables.save_results_table(df)
