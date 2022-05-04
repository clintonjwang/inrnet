import os, pdb, torch
osp = os.path
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
    pretrained = net_args['pretrained']
    # if net_args["type"] == "inr-effnet-b0":
    #     base = torchvision.models.efficientnet_b0(pretrained=pretrained)
    # elif net_args["type"] == "effnet-b0":
    #     return torchvision.models.efficientnet_b0(pretrained=pretrained)

    if net_args["type"] == "effnet-s3":
        m = torchvision.models.efficientnet_b0(pretrained=pretrained)
        return nn.Sequential(m.features[:3],
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(1),
            nn.Linear(24, args["data loading"]['classes']))

    elif net_args["type"] == "inr-effnet-s3":
        m = torchvision.models.efficientnet_b0(pretrained=pretrained)
        base = nn.Sequential(m.features[:3],
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Linear(24, args["data loading"]['classes']))

    else:
        raise NotImplementedError
        
    img_shape = args["data loading"]["image shape"]
    InrNet, _ = inn.conversion.translate_discrete_model(base, img_shape)
    return InrNet

def load_classifier_from_job(args):
    net_args = args["network"]
    net_args["training job"]
    img_shape = args["data loading"]["image shape"]
    InrNet, _ = inn.conversion.translate_discrete_model(base, img_shape)
    return InrNet


def finetune_classifier(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    global_step = 0
    loss_tracker = util.MetricTracker("loss", function=nn.CrossEntropyLoss())
    top5 = lambda labels, pred_cls: (labels.unsqueeze(1) == pred_cls).amax(1).float().mean()
    top1 = lambda labels, pred_cls: (labels.unsqueeze(1) == pred_cls[:,0]).float().mean()
    top5_tracker = util.MetricTracker("top5", function=top5)
    top1_tracker = util.MetricTracker("top1", function=top1)
    bsz = dl_args['batch size']

    def backprop():
        loss = loss_tracker(logits, labels)
        pred_cls = logits.topk(k=5).indices
        top_5 = top5_tracker(labels,pred_cls).item()
        top_1 = top1_tracker(labels,pred_cls).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if global_step % 20 == 0:
            print(np.round(loss.item(), decimals=3), "; top_5:", np.round(top_5, decimals=2),
                "; top_1:", np.round(top_1, decimals=2),
                flush=True)
        if global_step % 100 == 0:
            torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "best.pth"))
            loss_tracker.plot_running_average(path=paths["job output dir"]+"/plots/loss.png")
            top1_tracker.plot_running_average(path=paths["job output dir"]+"/plots/top1.png")
            top5_tracker.plot_running_average(path=paths["job output dir"]+"/plots/top5.png")
        # if attr_tracker.is_at_min("train"):
        #     torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "best.pth"))

    if args["network"]['type'].startswith('inr'):
        N = dl_args["initial sample points"]
        InrNet = load_pretrained_classifier(args)
        optimizer = torch.optim.Adam(InrNet.parameters(), lr=args["optimizer"]["learning rate"])
        for img_inr, labels in data_loader:
            global_step += 1
            img_inr = to_black_box(img_inr)
            logit_fxn = InrNet(img_inr)
            coords = logit_fxn.generate_sample_points(sample_size=N)
            logits = logit_fxn(coords)
            backprop()
            if global_step > args["optimizer"]["max steps"]:
                break

        torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "final.pth"))


    else:
        EffNet = load_pretrained_classifier(args).cuda()
        optimizer = torch.optim.Adam(EffNet.parameters(), lr=args["optimizer"]["learning rate"])
        for img_inr, labels in data_loader:
            global_step += 1
            img = to_black_box(img_inr).produce_images(*dl_args['image shape'])
            logits = EffNet(img)
            backprop()
            if global_step > args["optimizer"]["max steps"]:
                break

        torch.save(EffNet.state_dict(), osp.join(paths["weights dir"], "final.pth"))


def test_inr_classifier(args):
    inet_classes = dataloader.get_imagenet_classes()
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    InrNet = load_classifier_from_job(args)
    top_5, top_1 = 0,0
    dtop_5, dtop_1 = 0,0
    n_imgs = 0
    n_points = dl_args["validation sample points"]
    H,W = dl_args['image shape']
    # base = torchvision.models.efficientnet_b0(pretrained=True)
    # discrete_model = base.cuda().eval()
    # InrNet, output_shape = inn.conversion.translate_discrete_model(discrete_model, (H,W))
    for img_inr, class_ix in data_loader:
        n_imgs += 1
        with torch.no_grad():
            img_inr = to_black_box(img_inr)
            out_inr = InrNet(img_inr)
            out_inr.toggle_grid_mode(True)
            coords = out_inr.generate_sample_points(method="grid", dims=(H,W))
            # coords = out_inr.generate_sample_points(sample_size=n_points)
            logits = out_inr.eval()(coords).float()
            pred_cls = logits.topk(k=5).indices.cpu()
            top_5 += class_ix in pred_cls
            top_1 += (class_ix == pred_cls[0,0]).item()

            torch.cuda.empty_cache()
            x = img_inr.produce_image(H,W)#, split=2)
            dlogits = discrete_model(x)
            pred_cls = dlogits.topk(k=5).indices.cpu()
            dtop_5 += class_ix in pred_cls
            dtop_1 += (class_ix == pred_cls[0,0]).item()
            # if output_shape is not None:
            #     split = W // output_shape[-1]
            #     coords = util.first_split_meshgrid(H,W, split=split)
            #     logits = util.realign_values(logits, coords_gt=coords, inr=out_inr)
            #     logits = logits.reshape(*output_shape,-1).permute(2,0,1).unsqueeze(0)
            
            # if torch.allclose(dlogits, logits):#, rtol=.01, atol=.01):
            #     print('success')
            #     pdb.set_trace()
            #     return
            # else:
            #     #dlogits - logits
            #     pdb.set_trace()

        if n_imgs % 10 == 9:
            print("top_5:", top_5/n_imgs, "; top_1:", top_1/n_imgs,
                "; dtop_5:", dtop_5/n_imgs, "; dtop_1:", dtop_1/n_imgs)
        if n_imgs % 100 == 99:
            torch.save({"top 5": top_5, "top 1": top_1, "discrete top 5": dtop_5, "discrete top 1": dtop_1,
                "N":n_imgs}, paths["job output dir"]+"/results.txt")

