import os, pdb
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

def test_cyclegan(args):
    return
    
def train_cyclegan(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    horse_loader, zebra_loader = dataloader.get_inr_dataloader(dl_args)

    global_step = 0
    scaler = torch.cuda.amp.GradScaler()
    G_A2B = getGenerator(args)
    G_B2A = getGenerator(args)
    D_B = getDiscriminator(args)
    D_A = getDiscriminator(args)
    G_optim = torch.optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()),
        lr=args["optimizer"]["G learning rate"], betas=(0.5, 0.999))
    D_B_optim = torch.optim.Adam(D_B.parameters(),
        lr=args["optimizer"]["D learning rate"], betas=(0.5, 0.999))
    D_A_optim = torch.optim.Adam(D_A.parameters(),
        lr=args["optimizer"]["D learning rate"], betas=(0.5, 0.999))
    G_fxn, D_fxn = losses.adv_loss_fxns(args["loss settings"])

    def backward(loss, optimizer):
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    H,W = dl_args["image shape"]
    N = dl_args["initial sample points"]
    cycle_tracker = util.MetricTracker("cycle loss", function=losses.L1_dist_inr(N=N))
    ID_tracker = util.MetricTracker("ID loss", function=losses.L1_dist_inr(N=N))
    Gadv_tracker = util.MetricTracker("G adversary loss", function=G_fxn)
    Dadv_tracker = util.MetricTracker("D loss", function=D_fxn)
    metric_trackers = [cycle_tracker, Gadv_tracker, Dadv_tracker]
    for horse_inr in horse_loader:
        global_step += 1
        if global_step == N:
            N *= 2
            cycle_tracker.function = ID_tracker.function = losses.L1_dist_inr(N=N)
        true_zebra = to_black_box(next(zebra_loader), sample_size=N).cuda()
        true_horse = to_black_box(horse_inr, sample_size=N).cuda()
        with torch.cuda.amp.autocast():
            fake_horse = G_A2B(true_zebra)
            cycle_zebra = G_B2A(fake_horse)
            G_loss = cycle_tracker(cycle_zebra, true_zebra) * 10.
            G_loss += ID_tracker(G_B2A(true_zebra), true_zebra) * 5.

            fake_logit = D_B(fake_horse)
            G_loss += Gadv_tracker(fake_logit)
            backward(G_loss, G_optim)
            del G_loss, fake_logit

            true_logit = D_B(true_horse)
            fake_logit = D_B(G_A2B(true_zebra).detach())
            D_loss = Dadv_tracker(fake_logit, true_logit)
            backward(D_loss, D_B_optim)
            del D_loss, fake_logit, true_logit

            fake_zebra = G_B2A(true_horse)
            cycle_horse = G_A2B(fake_zebra)
            G_loss = cycle_tracker(cycle_horse, true_horse) * 10.
            G_loss += ID_tracker(G_A2B(true_horse), true_horse) * 5.

            fake_logit = D_A(fake_zebra)
            G_loss += Gadv_tracker(fake_logit)
            backward(G_loss, G_optim)
            del G_loss, fake_logit

            true_logit = D_A(true_zebra)
            fake_logit = D_A(G_B2A(true_horse).detach())
            D_loss = Dadv_tracker(fake_logit, true_logit)
            backward(D_loss, D_A_optim)
            del D_loss, fake_logit, true_logit


        if global_step % 10 == 0:
            print(Dadv_tracker.current_moving_average(20),
                Gadv_tracker.current_moving_average(20),
                ID_tracker.current_moving_average(20),
                cycle_tracker.current_moving_average(20),flush=True)
        if global_step % 20 == 0:
            torch.cuda.empty_cache()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    h,w = H//2, W//2
                    xy_grid = util.meshgrid_coords(h,w)

                    rgb = fake_horse.eval()(xy_grid)
                    rgb = util.realign_values(rgb, coords_gt=xy_grid, inr=fake_horse)
                    rgb = rescale_float(rgb.reshape(h,w,3).cpu().float().numpy())
                    plt.imsave(osp.join(paths["job output dir"]+"/imgs", f"{global_step//10}_horse.png"), rgb)

                    rgb = fake_zebra.eval()(xy_grid)
                    rgb = util.realign_values(rgb, coords_gt=xy_grid, inr=fake_zebra)
                    rgb = rescale_float(rgb.reshape(h,w,3).cpu().float().numpy())
                    plt.imsave(osp.join(paths["job output dir"]+"/imgs", f"{global_step//10}_zebra.png"), rgb)

            # torch.save(G_A2B.state_dict(), osp.join(paths["weights dir"], "netG_A2B.pth"))
            # torch.save(G_B2A.state_dict(), osp.join(paths["weights dir"], "netG_B2A.pth"))

        if global_step > args["optimizer"]["max steps"]:
            break

    # torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "final.pth"))

def getGenerator(args):
    net_args=args["network"]
    kwargs = dict(in_channels=3, out_channels=3, spatial_dim=2, dropout=net_args["dropout"])
    if net_args["G type"] == "UNet":
        model = inn.nets.UNet(min_channels=net_args["min channels"], **kwargs)
    elif net_args["G type"] == "ConvCM":
        model = inn.nets.ConvCM(min_channels=net_args["min channels"], **kwargs)
    elif net_args["G type"] == "FPN":
        model = inn.nets.FPN(min_channels=net_args["min channels"], **kwargs)
    else:
        raise NotImplementedError
    return model.cuda()

def getDiscriminator(args):
    net_args=args["network"]
    kwargs = dict(in_channels=3, out_channels=1, spatial_dim=2, activation=net_args["activation"], 
        final_activation=net_args["final activation"], dropout=net_args["dropout"])
    if net_args["D type"] == "Conv4":
        model = inn.nets.Conv4(min_channels=net_args["min channels"], **kwargs)
    elif net_args["D type"] == "ResNet":
        model = inn.nets.ResNet(min_channels=net_args["min channels"], **kwargs)
    else:
        raise NotImplementedError
    return model.cuda()
