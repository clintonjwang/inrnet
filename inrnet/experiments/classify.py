import os, pdb, torch
osp = os.path
import torch
nn = torch.nn
F = nn.functional
import monai.transforms as mtr
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models

from inrnet.data import dataloader
from inrnet import inn, util, losses
from inrnet.models.inrs.siren import to_black_box

rescale_clip = mtr.ScaleIntensityRangePercentiles(lower=5, upper=95, b_min=0, b_max=255, clip=True, dtype=np.uint8)
rescale_float = mtr.ScaleIntensity()

def load_pretrained_classifier(args):
    net_args = args["network"]
    if net_args["type"] == "effnet-b0":
        base = torchvision.models.efficientnet_b0(pretrained=True)
    elif net_args["type"] == "effnet-b1":
        base = torchvision.models.efficientnet_b1(pretrained=True)
    elif net_args["type"] == "effnet-b2":
        base = torchvision.models.efficientnet_b2(pretrained=True)
    else:
        raise NotImplementedError
    img_shape = args["data loading"]["image shape"]
    InrNet, _ = inn.conversion.translate_discrete_model(base, img_shape)
    return InrNet

def test_classifier(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    InrNet = load_pretrained_classifier(args)
    top_5, top_1 = 0,0
    n_imgs = 0
    n_points = args["data loading"]["validation sample points"]
    for img_inr, class_ix in data_loader:
        n_imgs += 1
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                img_inr = to_black_box(img_inr)
                output = InrNet(img_inr)
                coords = output.generate_sample_points(sample_size=n_points)
                pred_cls = output(coords).topk(k=5).indices.cpu()
                top_5 += class_ix in pred_cls
                top_1 += (class_ix == pred_cls[0,0]).item()

        if n_imgs % 10 == 9:
            print("top_5:", top_5/n_imgs, "; top_1:", top_1/n_imgs)
    torch.save({"top 5": top_5, "top_1": top_1, "N":n_imgs}, paths["job output dir"]+"/results.txt")


def finetune_classifier(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    # torch.autograd.set_detect_anomaly(True)

    global_step = 0
    scaler = torch.cuda.amp.GradScaler()
    InrNet = load_pretrained_classifier(args)
    optimizer = torch.optim.Adam(InrNet.parameters(), lr=args["optimizer"]["learning rate"])
    loss_tracker = util.MetricTracker("loss", function=losses.CrossEntropy())
    top_5, top_1 = 0,0
    for img_inr, class_ix in data_loader:
        global_step += 1
        img_inr = to_black_box(img_inr)
        with torch.cuda.amp.autocast():
            pred_cls = InrNet(img_inr)
            loss = loss_tracker(pred_cls, class_ix.cuda())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if global_step % 20 == 0:
            print(loss.item(),flush=True)
        if global_step % 100 == 0:
            torch.cuda.empty_cache()
            torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "best.pth"))

            # loss_tracker.plot_running_average(root=paths["job output dir"]+"/plots")

        # if attr_tracker.is_at_min("train"):
        #     torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "best.pth"))

        if global_step > args["optimizer"]["max steps"]:
            break

    torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "final.pth"))

def preprocess(xyz, global_step):
    if global_step < 1000:
        xyz = inn.functional.subsample_points_by_grid(xyz, spacing=.08)
    elif global_step < 5000:
        xyz = inn.functional.subsample_points_by_grid(xyz, spacing=.04)
    elif global_step < 9000:
        xyz = inn.functional.subsample_points_by_grid(xyz, spacing=.02)
    else:
        xyz = inn.functional.subsample_points_by_grid(xyz, spacing=.01)
    return xyz.half()
