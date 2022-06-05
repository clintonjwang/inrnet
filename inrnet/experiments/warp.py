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
from monai.networks.blocks import Warp, DVF2DDF

rescale_float = mtr.ScaleIntensity()
DS_DIR = "/data/vision/polina/scratch/clintonw/datasets"

def get_model(args):
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

def load_model_from_job(origin):
    orig_args = job_mgmt.get_job_args(origin)
    path = osp.expanduser(f"~/code/diffcoord/results/{origin}/weights/model.pth")
    model = get_model(orig_args)
    model.load_state_dict(torch.load(path))
    return model

def mean_l1(pred_seg, gt_seg):
    # pred_seg [B*N], gt_seg [B*N,C]
    iou_per_channel = (pred_seg & gt_seg).sum(0) / (pred_seg | gt_seg).sum(0)
    return iou_per_channel.mean()

# def dice(pred_seg, gt_seg):
#     return (pred_seg & gt_seg).sum() / pred_seg.size(0)

def get_seg_at_coords(seg, coords):
    coo = torch.floor(coords).long()
    return seg[...,coo[:,0], coo[:,1]].transpose(1,2)

def train_warp(args):
    vf2df = DVF2DDF().cuda()
    warp = Warp().cuda()
    paths = args["paths"]
    dl_args = args["data loading"]
    global_step = 0
    loss_tracker = util.MetricTracker("loss", function=nn.L1Loss())
    if dl_args['sample type'] == 'valid':
        dl_args['batch size'] = 1 #cannot handle different masks per datapoint
    data_loader = dataloader.get_inr_dataloader(dl_args)

    model = get_model(args).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["optimizer"]["learning rate"])
    for img_inrs, segs in data_loader:
        global_step += 1
        
        if args["network"]['type'].startswith('inr'):
            # if dl_args['sample type'] == 'grid':
            #     warp_inrs.change_sample_mode('grid')
            #     seg_gt = segs.flatten(start_dim=2).transpose(2,1)
            #     coords = warp_inrs.generate_sample_points(dims=dl_args['image shape'])
            # else:
            #     seg_gt = get_seg_at_coords(segs, coords)
            #     coords = warp_inrs.generate_sample_points(sample_size=dl_args["sample points"], method=dl_args['sample type'])
            # logits = warp_inrs.cuda()(coords)
            # if dl_args['sample type'] == 'grid':
            #     logits = util.realign_values(logits, coords=coords)
            imgs = img_inrs.produce_images(*dl_args['image shape'])
            src_imgs = imgs[:,:1]
            targ_imgs = imgs[:,1:]

            warp_inrs = model(img_inrs)
            warp_inrs.change_sample_mode('grid')
            grid_coords = warp_inrs.generate_sample_points(dims=dl_args['image shape'])
            VF_est = warp_inrs(grid_coords) #S->T
            VF_est = util.realign_values(VF_est, coords=grid_coords)
            VF_est = VF_est.transpose(2,1).reshape(-1,2,*dl_args['image shape'])
            DF_est = vf2df(-VF_est) #T->S
            est_imgs = warp(targ_imgs, DF_est)

        else:
            imgs = img_inrs.produce_images(*dl_args['image shape'])
            src_imgs = imgs[:,:1]
            targ_imgs = imgs[:,1:]

            VF_est = model(imgs)
            DF_est = vf2df(-VF_est)
            est_imgs = warp(targ_imgs, DF_est)
            # .flatten(start_dim=2).transpose(2,1)
            # seg_gt = segs.flatten(start_dim=2).transpose(2,1)

        # src_segs = segs[:,:1]
        # targ_segs = segs[:,1:]
        # est_segs = warp(targ_segs, DF_est)
        loss = loss_tracker(src_imgs, est_imgs)
        if torch.isnan(loss):
            raise ValueError('nan loss')

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if global_step % 20 == 0:
            print(np.round(loss.item(), decimals=3), flush=True)

        if global_step % 100 == 0:
            torch.save(model.state_dict(), osp.join(paths["weights dir"], "model.pth"))
            loss_tracker.plot_running_average(path=paths["job output dir"]+"/plots/loss.png")

            path = paths["job output dir"]+f"/imgs/{global_step}.png"
            save_example_imgs(path, targ_imgs[0,0], est_imgs[0,0], src_imgs[0,0])

        if global_step >= args["optimizer"]["max steps"]:
            break

    torch.save(model.state_dict(), osp.join(paths["weights dir"], "model.pth"))

# def compose_velocity_fields(warp_inrs, dl_args, vf1, vf2, vf3):
#     grid_coords = warp_inrs.generate_sample_points(dims=dl_args['image shape'])
#     VF_est = model(imgs)(grid_coords) #S->T
#     DF_est = vf2df(-VF_est) #T->S
#     est_imgs = warp(targ_imgs, DF_est)
#     est_segs = warp(targ_segs, DF_est)

def save_example_imgs(path, start, pred, end):
    coll_img = torch.cat((start, pred, end), dim=1).detach().cpu().numpy()
    plt.imsave(path, coll_img)

def test_inr_warp(args):
    vf2df = DVF2DDF().cuda()
    warp = Warp().cuda()
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    dices = []
    origin = args['target_job']
    model = load_model_from_job(origin).cuda().eval()
    orig_args = job_mgmt.get_job_args(origin)

    with torch.no_grad():
        for img_inrs, labels in data_loader:
            if orig_args["network"]['type'].startswith('inr'):
                imgs = img_inrs.produce_images(*dl_args['image shape'])
                src_imgs = imgs[:,:1]
                targ_imgs = imgs[:,1:]

                warp_inrs = model(img_inrs)
                warp_inrs.change_sample_mode('grid')
                grid_coords = warp_inrs.generate_sample_points(dims=dl_args['image shape'])
                VF_est = warp_inrs(grid_coords) #S->T
                VF_est = util.realign_values(VF_est, coords=grid_coords)
                VF_est = VF_est.transpose(2,1).reshape(-1,2,*dl_args['image shape'])
                DF_est = vf2df(-VF_est) #T->S
                est_imgs = warp(targ_imgs, DF_est)
                
                dices.append()
            else:
                img = img_inrs.produce_images(*dl_args['image shape'])
                logits = model(img)
                pred_cls = logits.topk(k=3).indices
                dices.append()

    torch.save((dices,), osp.join(paths["job output dir"], "stats.pt"))
