import os, pdb, torch
osp = os.path
nn = torch.nn
F = nn.functional
import monai.transforms as mtr
import numpy as np
import matplotlib.pyplot as plt

from inrnet.data import dataloader
from inrnet import inn, util, losses, jobs as job_mgmt
import inrnet.inn.nets.convnext
import inrnet.models.convnext

rescale_clip = mtr.ScaleIntensityRangePercentiles(lower=5, upper=95, b_min=0, b_max=255, clip=True, dtype=np.uint8)
rescale_float = mtr.ScaleIntensity()

def load_pretrained_model(args):
    net_args = args["network"]
    pretrained = net_args['pretrained']
    if isinstance(pretrained, str):
        raise NotImplementedError
        base = load_model_from_job(pretrained)
    else:
        if net_args["type"] == "convnext":
            return inrnet.models.convnext.mini_convnext()
        elif net_args["type"] == "inr-convnext":
            return inrnet.inn.nets.convnext.translate_convnext_model(args["data loading"]["image shape"])
        else:
            raise NotImplementedError

def load_model_from_job(origin):
    orig_args = job_mgmt.get_job_args(origin)
    path = osp.expanduser(f"~/code/diffcoord/results/{origin}/weights/best.pth")
    model = load_pretrained_model(orig_args)
    model.load_state_dict(torch.load(path))
    return model

def pixel_acc(pred_seg, gt_seg):
    return (pred_seg == gt_seg).float().mean()

def finetune_segmenter(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    global_step = 0
    loss_tracker = util.MetricTracker("loss", function=nn.CrossEntropyLoss())
    iou = lambda pred_cls, labels: (labels.unsqueeze(1) == pred_cls).amax(1).float().mean()
    acc = lambda pred_cls, labels: (labels == pred_cls[:,0]).float().mean()
    iou_tracker = util.MetricTracker("IoU", function=iou)
    acc_tracker = util.MetricTracker("acc", function=acc)
    bsz = dl_args['batch size']

    def backprop(network):
        loss = loss_tracker(logits, labels)
        pred_seg = logits.max(1).indices
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
            iou_tracker.plot_running_average(path=paths["job output dir"]+"/plots/iou.png")
            acc_tracker.plot_running_average(path=paths["job output dir"]+"/plots/acc.png")
        # if attr_tracker.is_at_min("train"):
        #     torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "best.pth"))

        # img_inr = to_black_box(img_inr)
        # Seg_inr = InrNet(img_inr)
        # seg_pred = Seg_inr(xyz[0,:,:2])
        # loss = loss_tracker(z_pred, xyz[0,:,-1])
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # if global_step % 10 == 0:
        #     print(loss.item(),flush=True)
        #     del loss, z_pred
        #     torch.cuda.empty_cache()
        #     with torch.no_grad():
        #         with torch.cuda.amp.autocast():
        #             h,w = H//4, W//4
        #             tensors = [torch.linspace(-1, 1, steps=h), torch.linspace(-1, 1, steps=w)]
        #             mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
        #             xy_grid = mgrid.reshape(-1, 2).half().cuda()
        #             InrNet.eval()
        #             z_pred = Seg_inr(xy_grid)
        #             z_pred = rescale_float(z_pred.reshape(h,w).cpu().float().numpy())
        #             plt.imsave(osp.join(paths["job output dir"]+"/imgs", f"{global_step}_z.png"), z_pred, cmap="gray")

        #             del z_pred, Seg_inr
        #             torch.cuda.empty_cache()
        #             rgb = img_inr.evaluator(xy_grid)
        #             rgb = rescale_float(rgb.reshape(h,w, 3).cpu().float().numpy())
        #             plt.imsave(osp.join(paths["job output dir"]+"/imgs", f"{global_step}_rgb.png"), rgb)

        #             torch.cuda.empty_cache()
        #             InrNet.train()

    model = load_pretrained_model(args).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["optimizer"]["learning rate"])
    if args["network"]['type'].startswith('inr'):
        N = dl_args["sample points"]
        for img_inr, seg in data_loader:
            global_step += 1
            seg_inr = model(img_inr)
            coords = seg_inr.generate_sample_points(sample_size=N, method='rqmc')
            pdb.set_trace()
            logits = seg_inr.cuda()(coords)
            backprop(model)
            if global_step >= args["optimizer"]["max steps"]:
                break

    else:
        for img_inr, seg in data_loader:
            global_step += 1
            img = img_inr.produce_images(*dl_args['image shape'])
            logits = model(img)
            backprop(model)
            if global_step >= args["optimizer"]["max steps"]:
                break

    torch.save(model.state_dict(), osp.join(paths["weights dir"], "final.pth"))


def test_inr_classifier(args):
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
