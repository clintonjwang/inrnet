import os, pdb, torch
osp = os.path
nn = torch.nn
F = nn.functional
import monai.transforms as mtr
import numpy as np
import matplotlib.pyplot as plt

from inrnet.data import dataloader
from inrnet import inn, util, losses, jobs as job_mgmt
from inrnet.inn import qmc, functional as inrF
import inrnet.inn.nets.convnext
import inrnet.models.convnext
import inrnet.models.common
import inn.nets.inr2inr

rescale_float = mtr.ScaleIntensity()
DS_DIR = "/data/vision/polina/scratch/clintonw/datasets"

def load_pretrained_model(args):
    net_args = args["network"]
    if net_args["type"] == "inr-3":
        return inn.nets.inr2inr.ISeg3(in_channels=2, out_channels=2)
    elif net_args["type"] == "inr-5":
        return inn.nets.inr2inr.ISeg5(in_channels=2, out_channels=2)
    elif net_args["type"] == "cnn-3":
        return inrnet.models.common.Seg3(in_channels=2, out_channels=2)
    elif net_args["type"] == "cnn-5":
        return inrnet.models.common.Seg5(in_channels=2, out_channels=2)
    else:
        raise NotImplementedError
    return InrNet

def load_model_from_job(origin):
    orig_args = job_mgmt.get_job_args(origin)
    path = osp.expanduser(f"~/code/diffcoord/results/{origin}/weights/model.pth")
    model = load_pretrained_model(orig_args)
    model.load_state_dict(torch.load(path))
    return model

def mean_l1(pred_atlas, gt_atlas):
    # pred_atlas [B*N], gt_atlas [B*N,C]
    iou_per_channel = (pred_atlas & gt_atlas).sum(0) / (pred_atlas | gt_atlas).sum(0)
    return iou_per_channel.mean()

# def dice(pred_atlas, gt_atlas):
#     return (pred_atlas & gt_atlas).sum() / pred_atlas.size(0)

def get_atlas_at_coords(atlas, coords):
    coo = torch.floor(coords).long()
    return atlas[...,coo[:,0], coo[:,1]].transpose(1,2)

def train_warp(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    global_step = 0
    loss_tracker = util.MetricTracker("loss", function=nn.L1Loss())
    if dl_args['sample type'] == 'valid':
        dl_args['batch size'] = 1 #cannot handle different masks per datapoint
    data_loader = dataloader.get_inr_dataloader(dl_args)

    model = load_pretrained_model(args).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["optimizer"]["learning rate"])
    for img_inr, atlass in data_loader:
        global_step += 1
        
        if args["network"]['type'].startswith('inr'):
            # if global_step == args["optimizer"]["max steps"]//2:
            #     inrnet.inn.nets.convnext.enable_cn_blocks(model)
            atlas_inr = model(img_inr)
            if dl_args['sample type'] == 'grid':
                atlas_inr.change_sample_mode('grid')
                atlas_gt = atlass.flatten(start_dim=2).transpose(2,1)
                coords = atlas_inr.generate_sample_points(dims=dl_args['image shape'])
            else:
                atlas_gt = get_atlas_at_coords(atlass, coords)
                coords = atlas_inr.generate_sample_points(sample_size=dl_args["sample points"], method=dl_args['sample type'])
            logits = atlas_inr.cuda()(coords)
            if dl_args['sample type'] == 'grid':
                logits = util.realign_values(logits, coords=coords)

        else:
            img = img_inr.produce_images(*dl_args['image shape'])
            logits = model(img).flatten(start_dim=2).transpose(2,1)
            atlas_gt = atlass.flatten(start_dim=2).transpose(2,1)

        # loss = loss_tracker(logits, atlas_gt.float()) #BCE
        maxes, gt_labels = atlas_gt.max(-1)
        loss = loss_tracker(logits[maxes != 0], gt_labels[maxes != 0]) #cross entropy
        if torch.isnan(loss):
            raise ValueError('nan loss')
        pred_atlas = logits.max(-1).indices
        pred_1hot = F.one_hot(pred_atlas, num_classes=atlas_gt.size(-1)).bool()
        iou = iou_tracker(pred_1hot[maxes != 0], atlas_gt[maxes != 0]).item()
        acc = acc_tracker(pred_1hot[maxes != 0], atlas_gt[maxes != 0]).item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if global_step % 20 == 0:
            print(np.round(loss.item(), decimals=3), "; iou:", np.round(iou, decimals=3),
                "; acc:", np.round(acc*100, decimals=2),
                flush=True)

        if global_step % 100 == 0:
            torch.save(model.state_dict(), osp.join(paths["weights dir"], "model.pth"))
            loss_tracker.plot_running_average(path=paths["job output dir"]+"/plots/loss.png")
            iou_tracker.plot_running_average(path=paths["job output dir"]+"/plots/iou.png")
            acc_tracker.plot_running_average(path=paths["job output dir"]+"/plots/acc.png")

            path = paths["job output dir"]+f"/imgs/{global_step}.png"
            with torch.no_grad():
                rgb = img_inr.produce_images(*dl_args['image shape'])[0]
                if args["network"]['type'].startswith('inr') and dl_args['sample type'] == 'masked':
                    # grid_atlas = torch.zeros(*dl_args['image shape'], device=coords.device, dtype=torch.long)
                    # coo = torch.floor(coords).long()
                    # grid_atlas[coo[:,0], coo[:,1]] = pred_atlas[0]

                    # gt_label = atlass[0].max(0).indices
                    # save_example_atlass(path, rgb, grid_atlas, gt_label)
                    
                    atlas_gt = atlass.flatten(start_dim=2).transpose(2,1)
                    maxes, gt_labels = atlas_gt.max(-1)
                    atlas_inr = model(img_inr)
                    atlas_inr.change_sample_mode('grid')
                    coords = atlas_inr.generate_sample_points(dims=dl_args['image shape'])
                    logits = atlas_inr.cuda()(coords)
                    logits = util.realign_values(logits, coords=coords)
                    pred_atlas = logits.max(-1).indices
                    gt_labels += 1
                    gt_labels[maxes == 0] = 0
                    pred_atlas += 1
                    pred_atlas[maxes == 0] = 0
                    save_example_atlass(path, rgb, pred_atlas[0].reshape(*rgb.shape[-2:]), gt_labels[0].reshape(*rgb.shape[-2:]))

                else:
                    gt_labels += 1
                    gt_labels[maxes == 0] = 0
                    pred_atlas += 1
                    pred_atlas[maxes == 0] = 0
                    save_example_atlass(path, rgb, pred_atlas[0].reshape(*rgb.shape[-2:]), gt_labels[0].reshape(*rgb.shape[-2:]))

        if global_step >= args["optimizer"]["max steps"]:
            break

    torch.save(model.state_dict(), osp.join(paths["weights dir"], "model.pth"))


def test_inr_warp(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    top3, top1 = 0,0
    origin = args['target_job']
    model = load_model_from_job(origin).cuda().eval()
    orig_args = job_mgmt.get_job_args(origin)

    if orig_args["network"]['type'].startswith('inr'):
        N = dl_args["sample points"]
        for img_inr, labels in data_loader:
            with torch.no_grad():
                logit_fxn = model(img_inr).cuda().eval()
                coords = logit_fxn.generate_sample_points(sample_size=N)
                logits = logit_fxn(coords)
                pred_cls = logits.topk(k=3).indices
                top3 += (labels.unsqueeze(1) == pred_cls).amax(1).float().sum().item()
                top1 += (labels == pred_cls[:,0]).float().sum().item()

    else:
        for img_inr, labels in data_loader:
            with torch.no_grad():
                img = img_inr.produce_images(*dl_args['image shape'])
                logits = model(img)
                pred_cls = logits.topk(k=3).indices
                top3 += (labels.unsqueeze(1) == pred_cls).amax(1).float().sum().item()
                top1 += (labels == pred_cls[:,0]).float().sum().item()

    torch.save((top1, top3), osp.join(paths["job output dir"], "stats.pt"))
