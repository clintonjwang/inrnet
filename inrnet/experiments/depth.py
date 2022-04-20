import os, pdb, torch
osp = os.path
import torch
nn = torch.nn
F = nn.functional
import monai.transforms as mtr
import numpy as np
import matplotlib.pyplot as plt

from inrnet.data import dataloader
from inrnet import inn, util, losses
from inrnet.models.inrs.siren import to_black_box

rescale_clip = mtr.ScaleIntensityRangePercentiles(lower=5, upper=95, b_min=0, b_max=255, clip=True, dtype=np.uint8)
rescale_float = mtr.ScaleIntensity()

def train_depth_model(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)

    global_step = 0
    scaler = torch.cuda.amp.GradScaler()
    InrNet = getDepthNet(args)
    optimizer = torch.optim.Adam(InrNet.parameters(), lr=args["optimizer"]["learning rate"])
    H,W = dl_args["image shape"]
    loss_tracker = util.MetricTracker("loss", function=losses.L1_dist)
    for img_inr, xyz in data_loader:
        xyz.squeeze_()
        global_step += 1
        xyz[:,0] /= W/2
        xyz[:,1] /= H/2
        xyz[:,2] /= dl_args["depth scale"]
        xyz[:,:2] -= 1
        if global_step < 1000:
            xyz = inn.functional.subsample_points_by_grid(xyz, spacing=.08)
        elif global_step < 5000:
            xyz = inn.functional.subsample_points_by_grid(xyz, spacing=.04)
        elif global_step < 9000:
            xyz = inn.functional.subsample_points_by_grid(xyz, spacing=.02)
        else:
            xyz = inn.functional.subsample_points_by_grid(xyz, spacing=.01)
        xyz = xyz.half()
        img_inr = to_black_box(img_inr)
        with torch.cuda.amp.autocast():
            depth_inr = InrNet(img_inr)
            if "DEBUG" in args:
                loss = loss_tracker(inr=depth_inr, gt_values=img_inr(xyz[:,:2]).mean(-1), coords=xyz[:,:2])
            else:
                loss = loss_tracker(inr=depth_inr, gt_values=xyz[:,-1], coords=xyz[:,:2])
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if global_step % 1 == 0:
            print(loss.item(),flush=True)
        if global_step % 1 == 0:
            torch.cuda.empty_cache()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    h,w = H//2, W//2
                    z_pred = rescale_clip(depth_inr.eval().produce_image(h,w, split=2))
                    plt.imsave(osp.join(paths["job output dir"]+"/imgs", f"{global_step//10}_z.png"), z_pred, cmap="gray")
                    rgb = rescale_float(img_inr.produce_image(h,w))
                    plt.imsave(osp.join(paths["job output dir"]+"/imgs", f"{global_step//10}_rgb.png"), rgb)

            torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "best.pth"))

            # loss_tracker.plot_running_average(root=paths["job output dir"]+"/plots")

        # if attr_tracker.is_at_min("train"):
        #     torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "best.pth"))

        if global_step > args["optimizer"]["max steps"]:
            break

    torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "final.pth"))


def getDepthNet(args):
    net_args=args["network"]
    kwargs = dict(in_channels=3, out_channels=1, spatial_dim=2, activation=net_args["activation"], 
        final_activation=net_args["final activation"], dropout=net_args["dropout"])
    if net_args["type"] == "UNet":
        model = inn.nets.UNet(min_channels=net_args["min channels"], **kwargs)
    elif net_args["type"] == "ConvCM":
        model = inn.nets.ConvCM(min_channels=net_args["min channels"], **kwargs)
    elif net_args["type"] == "FPN":
        model = inn.nets.FPN(min_channels=net_args["min channels"], **kwargs)
    else:
        raise NotImplementedError("bad type: " + net_args["type"])

    #load_checkpoint(model, paths)
    return model.cuda()
