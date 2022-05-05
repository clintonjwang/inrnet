import os, pdb, torch
osp = os.path
nn = torch.nn
F = nn.functional
import monai.transforms as mtr
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models

from inrnet.data import dataloader
from inrnet import inn, util, losses, jobs as job_mgmt

rescale_clip = mtr.ScaleIntensityRangePercentiles(lower=5, upper=95, b_min=0, b_max=255, clip=True, dtype=np.uint8)
rescale_float = mtr.ScaleIntensity()

def load_pretrained_classifier(args):
    net_args = args["network"]
    pretrained = net_args['pretrained']
    # if net_args["type"] == "inr-effnet-b0":
    #     base = torchvision.models.efficientnet_b0(pretrained=pretrained)
    # elif net_args["type"] == "effnet-b0":
    #     return torchvision.models.efficientnet_b0(pretrained=pretrained)
    out = nn.Linear(24, args["data loading"]['classes'])
    nn.init.kaiming_uniform_(out.weight, mode='fan_in')
    out.bias.data.zero_()
    if net_args["type"] == "effnet-s3":
        m = torchvision.models.efficientnet_b0(pretrained=pretrained)
        return nn.Sequential(m.features[:3],
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(1), out)

    elif net_args["type"] == "inr-effnet-s3":
        m = torchvision.models.efficientnet_b0(pretrained=pretrained)
        base = nn.Sequential(m.features[:3],
            nn.AdaptiveAvgPool2d(output_size=1), out)

    else:
        raise NotImplementedError
        
    img_shape = args["data loading"]["image shape"]
    InrNet, _ = inn.conversion.translate_discrete_model(base, img_shape)
    return InrNet

def load_classifier_from_job(origin):
    orig_args = job_mgmt.get_job_args(origin)
    path = osp.expanduser(f"~/code/diffcoord/results/{origin}/weights/final.pth")
    model = load_pretrained_classifier(orig_args)
    model.load_state_dict(torch.load(path))
    return model


def finetune_classifier(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    global_step = 0
    loss_tracker = util.MetricTracker("loss", function=nn.CrossEntropyLoss())
    top5 = lambda pred_cls, labels: (labels.unsqueeze(1) == pred_cls).amax(1).float().mean()
    top1 = lambda pred_cls, labels: (labels == pred_cls[:,0]).float().mean()
    top5_tracker = util.MetricTracker("top5", function=top5)
    top1_tracker = util.MetricTracker("top1", function=top1)
    bsz = dl_args['batch size']

    def backprop(network):
        loss = loss_tracker(logits, labels)
        pred_cls = logits.topk(k=5).indices
        top_5 = top5_tracker(pred_cls, labels).item()
        top_1 = top1_tracker(pred_cls, labels).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if global_step % 20 == 0:
            print(np.round(loss.item(), decimals=3), "; top_5:", np.round(top_5, decimals=2),
                "; top_1:", np.round(top_1, decimals=2),
                flush=True)
        if global_step % 100 == 0:
            torch.save(network.state_dict(), osp.join(paths["weights dir"], "best.pth"))
            loss_tracker.plot_running_average(path=paths["job output dir"]+"/plots/loss.png")
            top1_tracker.plot_running_average(path=paths["job output dir"]+"/plots/top1.png")
            top5_tracker.plot_running_average(path=paths["job output dir"]+"/plots/top5.png")
        # if attr_tracker.is_at_min("train"):
        #     torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "best.pth"))

    if args["network"]['type'].startswith('inr'):
        N = dl_args["sample points"]
        InrNet = load_pretrained_classifier(args)
        optimizer = torch.optim.Adam(InrNet.parameters(), lr=args["optimizer"]["learning rate"])
        for img_inr, labels in data_loader:
            global_step += 1
            logit_fxn = InrNet(img_inr)
            coords = logit_fxn.generate_sample_points(sample_size=N, method='rqmc')
            logits = logit_fxn(coords)
            backprop(InrNet)
            if global_step > args["optimizer"]["max steps"]:
                break

        torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "final.pth"))


    else:
        EffNet = load_pretrained_classifier(args).cuda()
        optimizer = torch.optim.Adam(EffNet.parameters(), lr=args["optimizer"]["learning rate"])
        for img_inr, labels in data_loader:
            global_step += 1
            img = img_inr.produce_images(*dl_args['image shape'])
            logits = EffNet(img)
            backprop(EffNet)
            if global_step > args["optimizer"]["max steps"]:
                break

        torch.save(EffNet.state_dict(), osp.join(paths["weights dir"], "final.pth"))


def test_inr_classifier(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    top3, top1 = 0,0
    origin = args['target_job']
    model = load_classifier_from_job(origin).cuda().eval()
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
