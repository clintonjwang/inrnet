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

rescale_float = mtr.ScaleIntensity()

def train_depth_model(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)

    global_step = 0
    scaler = torch.cuda.amp.GradScaler()
    InrNet = getDepthNet(args).train()
    optimizer = torch.optim.Adam(InrNet.parameters(), lr=args["optimizer"]["learning rate"])
    H,W = dl_args["image shape"]
    loss_tracker = util.MetricTracker("loss", function=losses.L1_dist)
    for img_inr, xyz in data_loader:
        global_step += 1
        print(".",end="",flush=True)
        # xyz = subsample(xyz, dl_args["sample_fraction"])
        xyz[0,:,0] /= W/2
        xyz[0,:,1] /= H/2
        xyz[0,:,2] /= dl_args["depth scale"]
        xyz[0,:,:2] -= 1
        xyz = xyz.half()
        img_inr = to_black_box(img_inr)
        with torch.cuda.amp.autocast():
            depth_inr = InrNet(img_inr)
            z_pred = depth_inr(xyz[0,:,:2])
            loss = loss_tracker(z_pred, xyz[0,:,-1])
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()

        if global_step % 10 == 0:
            print(loss.item(),flush=True)
            del loss, z_pred
            torch.cuda.empty_cache()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    h,w = H//4, W//4
                    xy_grid = util.meshgrid_coords(h,w)
                    InrNet.eval()
                    z_pred = depth_inr(xy_grid)
                    z_pred = rescale_float(z_pred.reshape(h,w).cpu().float().numpy())
                    plt.imsave(osp.join(paths["job output dir"]+"/imgs", f"{global_step}_z.png"), z_pred, cmap="gray")

                    del z_pred, depth_inr
                    torch.cuda.empty_cache()
                    rgb = img_inr.evaluator(xy_grid)
                    rgb = rescale_float(rgb.reshape(h,w, 3).cpu().float().numpy())
                    plt.imsave(osp.join(paths["job output dir"]+"/imgs", f"{global_step}_rgb.png"), rgb)

                    torch.cuda.empty_cache()
                    InrNet.train()

            torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "best.pth"))

            # loss_tracker.plot_running_average(root=paths["job output dir"]+"/plots")
            # util.save_examples(global_step, paths["job output dir"]+"/imgs",
            #     example_outputs["orig_imgs"][:2], example_outputs["fake_img"][:2],
            #     example_outputs["recon_img"][:2], transforms=transforms)

        # if attr_tracker.is_at_min("train"):
        #     torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "best.pth"))

        if global_step > args["optimizer"]["max steps"]:
            break

    torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "final.pth"))

def getDepthNet(args):
    net_args=args["network"]
    kwargs = dict(in_channels=3, out_channels=1, spatial_dim=2, dropout=net_args["dropout"])
    if net_args["type"] == "ConvCM":
        model = inn.nets.ConvCM(mid_channels=net_args["min channels"], **kwargs)
    elif net_args["type"] == "Conv4":
        model = inn.nets.Conv4(mid_channels=net_args["min channels"], **kwargs)
    elif net_args["type"] == "UNet":
        model = inn.nets.UNet(min_channels=net_args["min channels"], **kwargs)
    elif net_args["type"] == "FPN":
        model = inn.nets.FPN(mid_channels=net_args["min channels"], **kwargs)
    elif net_args["type"] == "CmPlCm":
        model = inn.nets.CmPlCm(**kwargs)
    elif net_args["type"] == "ResNet":
        model = inn.nets.ResNet(**kwargs)
    else:
        raise NotImplementedError

    #load_checkpoint(model, paths)
    return model.cuda().eval()
