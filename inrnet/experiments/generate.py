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
        if net_args["type"] == "wgan":
            return inrnet.models.wgan.wgan()
        elif net_args["type"] == "inr-wgan":
            G,D = inrnet.inn.nets.wgan.translate_wgan_model()
        else:
            raise NotImplementedError
    return G.cuda(), D.cuda()

def load_model_from_job(origin):
    orig_args = job_mgmt.get_job_args(origin)
    path = osp.expanduser(f"~/code/diffcoord/results/{origin}/weights/best.pth")
    model = load_pretrained_model(orig_args)
    model.load_state_dict(torch.load(path))
    return model

def train_generator(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    global_step = 0
    loss_tracker = util.MetricTracker("loss", function=nn.BCEWithLogitsLoss())
    # fid_tracker = util.MetricTracker("FID", function=fid)
    bsz = dl_args['batch size']

    G,D = load_pretrained_model(args)
    G_optim = torch.optim.AdamW(G.parameters(), lr=args["optimizer"]["learning rate"])
    D_optim = torch.optim.AdamW(D.parameters(), lr=args["optimizer"]["learning rate"])
    for true_inr in data_loader:
        global_step += 1
        noise = torch.randn(dl_args["batch size"], 128, device='cuda')
        if args["network"]['type'].startswith('inr'):
            gen_inr = G(noise)
            if dl_args['sample type'] == 'qmc':
                coords = true_inr.generate_sample_points(sample_size=dl_args["sample points"], method='qmc')
            else:
                true_inr.toggle_grid_mode(True)
                coords = true_inr.generate_sample_points(dims=dl_args['image shape'])
            fake_logits = D(gen_inr)(coords)

        else:
            gen_img = G(noise)
            fake_logits = D(gen_img)

        G_loss = G_loss_tracker(fake_logits)
        G_optim.zero_grad(set_to_none=True)
        G_loss.backward()
        G_optim.step()

        if args["network"]['type'].startswith('inr'):
            fake_logits = D(gen_inr.detach())(coords)
            true_logits = D(true_inr)
        else:
            true_img = true_inr.produce_images(*dl_args['image shape'])
            fake_logits = D(gen_img.detach())
            true_logits = D(true_img)

        D_loss = D_loss_tracker(fake_logits, true_logits)
        D_optim.zero_grad(set_to_none=True)
        D_loss.backward()
        D_optim.step()

        if global_step % 20 == 0:
            print("G:", np.round(G_loss.item(), decimals=3),
                "; D:", np.round(D_loss, decimals=3), flush=True)
            save_examples()
        if global_step % 100 == 0:
            torch.save(network.state_dict(), osp.join(paths["weights dir"], "best.pth"))
            G_loss_tracker.plot_running_average(path=paths["job output dir"]+"/plots/G.png")
            D_loss_tracker.plot_running_average(path=paths["job output dir"]+"/plots/D.png")

        if global_step >= args["optimizer"]["max steps"]:
            break

    torch.save(model.state_dict(), osp.join(paths["weights dir"], "final.pth"))


def test_inr_generator(args):
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

def save_examples():
    return
    with torch.no_grad():
        h,w = H//4, W//4
        tensors = [torch.linspace(-1, 1, steps=h), torch.linspace(-1, 1, steps=w)]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
        xy_grid = mgrid.reshape(-1, 2).half().cuda()
        InrNet.eval()
        z_pred = Seg_inr(xy_grid)
        z_pred = rescale_float(z_pred.reshape(h,w).cpu().float().numpy())
        plt.imsave(osp.join(paths["job output dir"]+"/imgs", f"{global_step}_z.png"), z_pred, cmap="gray")

        del z_pred, Seg_inr
        torch.cuda.empty_cache()
        rgb = img_inr.evaluator(xy_grid)
        rgb = rescale_float(rgb.reshape(h,w, 3).cpu().float().numpy())
        plt.imsave(osp.join(paths["job output dir"]+"/imgs", f"{global_step}_rgb.png"), rgb)

        torch.cuda.empty_cache()
        InrNet.train()
        