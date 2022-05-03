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
    if net_args["type"] == "inr-effnet-b0":
        base = torchvision.models.efficientnet_b0(pretrained=True)
    elif net_args["type"] == "effnet-b0":
        return torchvision.models.efficientnet_b0(pretrained=True)
    elif net_args["type"] == "inr-effnet-b1":
        base = torchvision.models.efficientnet_b1(pretrained=True)
    elif net_args["type"] == "inr-effnet-b2":
        base = torchvision.models.efficientnet_b2(pretrained=True)
    elif net_args["type"] == "inr-effnet-s3":
        m = torchvision.models.efficientnet_b0(pretrained=True)
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
    scaler = torch.cuda.amp.GradScaler()
    loss_tracker = util.MetricTracker("loss", function=nn.CrossEntropyLoss())
    top5_tracker = util.MetricTracker("top5")
    top1_tracker = util.MetricTracker("top1")
    top_5, top_1 = 0,0

    if args["network"]['type'].startswith('inr'):
        N = dl_args["initial sample points"]
        InrNet = load_pretrained_classifier(args)
        optimizer = torch.optim.Adam(InrNet.parameters(), lr=args["optimizer"]["learning rate"])
        for img_inr, labels in data_loader:
            global_step += 1
            # if global_step == N//4:
            #     N *= 2
            img_inr = to_black_box(img_inr)
            pdb.set_trace()
            logit_fxn = InrNet(img_inr)
            coords = logit_fxn.generate_sample_points(sample_size=N)
            logits = logit_fxn(coords)
            pred_cls = logits.topk(k=5).indices.cpu()
            top_5 += labels in pred_cls
            top_1 += (labels == pred_cls[0,0]).item()
            loss = loss_tracker(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if global_step % 20 == 0:
                print(np.round(loss.item(), decimals=3), "; top_5:", np.round(top_5/global_step, decimals=3), "; top_1:",
                    np.round(top_1/global_step, decimals=3),
                    flush=True)
            if global_step % 100 == 0:
                torch.cuda.empty_cache()
                torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "best.pth"))
                loss_tracker.plot_running_average(path=paths["job output dir"]+"/plots/loss.png")
            # if attr_tracker.is_at_min("train"):
            #     torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "best.pth"))
            if global_step > args["optimizer"]["max steps"]:
                break

        torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "final.pth"))


    else:
        EffNet = load_pretrained_classifier(args).cuda()
        optimizer = torch.optim.Adam(EffNet.parameters(), lr=args["optimizer"]["learning rate"])
        for img_inr, class_ix in data_loader:
            global_step += 1
            img = to_black_box(img_inr).produce_image(*dl_args['image shape'], split=2, format='torch')
            logits = EffNet(img)
            pred_cls = logits.topk(k=5).indices.cpu()
            top_5 += class_ix in pred_cls
            top_1 += (class_ix == pred_cls[0,0]).item()
            loss = loss_tracker(logits, class_ix.cuda())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if global_step % 50 == 0:
                print(np.round(loss.item(), decimals=3), "; top_5:", np.round(top_5/global_step, decimals=3), "; top_1:",
                    np.round(top_1/global_step, decimals=3),
                    flush=True)
            if global_step % 500 == 0:
                torch.cuda.empty_cache()
                torch.save(EffNet.state_dict(), osp.join(paths["weights dir"], "best.pth"))
                loss_tracker.plot_running_average(path=paths["job output dir"]+"/plots/loss.png")
            # if attr_tracker.is_at_min("train"):
            #     torch.save(EffNet.state_dict(), osp.join(paths["weights dir"], "best.pth"))
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

