import os, pdb, torch
osp = os.path
nn = torch.nn
F = nn.functional
import monai.transforms as mtr
import numpy as np
import matplotlib.pyplot as plt

from inrnet.data import dataloader
from inrnet import inn, util, losses, jobs as job_mgmt
import inrnet.inn.nets.wgan
import inrnet.models.wgan

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
            G,D = inrnet.models.wgan.simple_wgan()
        elif net_args["type"] == "inr-wgan":
            G,D = inrnet.inn.nets.wgan.translate_wgan_model()
        else:
            raise NotImplementedError
    return G.cuda(), D.cuda()

def load_model_from_job(origin):
    orig_args = job_mgmt.get_job_args(origin)
    Gpath = osp.expanduser(f"~/code/diffcoord/results/{origin}/weights/G.pth")
    Dpath = osp.expanduser(f"~/code/diffcoord/results/{origin}/weights/D.pth")
    G,D = load_pretrained_model(orig_args)
    G.load_state_dict(torch.load(Gpath))
    D.load_state_dict(torch.load(Dpath))
    return G,D

def train_generator(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    global_step = 0
    G_fxn, D_fxn = losses.adv_loss_fxns(args["loss settings"])
    G_loss_tracker = util.MetricTracker("G loss", function=G_fxn)
    D_loss_tracker = util.MetricTracker("D loss", function=D_fxn)
    GP_tracker = util.MetricTracker("GP", function=losses.gradient_penalty)
    bsz = dl_args['batch size']

    G,D = load_pretrained_model(args)
    G_optim = torch.optim.AdamW(G.parameters(), lr=args["optimizer"]["learning rate"], betas=(.5,.999))
    D_optim = torch.optim.AdamW(D.parameters(), lr=args["optimizer"]["learning rate"], betas=(.5,.999))
    for true_inr in data_loader:
        global_step += 1
        noise = torch.randn(dl_args["batch size"], 128, device='cuda')
        if global_step % 4 == 0:
            if args["network"]['type'].startswith('inr'):
                gen_inr = G(noise)
                fake_fxn = D(gen_inr)
                if dl_args['sample type'] == 'qmc':
                    coords = true_inr.generate_sample_points(sample_size=dl_args["initial sample points"], method='qmc')
                else:
                    fake_fxn.toggle_grid_mode(True)
                    coords = true_inr.generate_sample_points(dims=dl_args['initial grid shape'], method='grid')
                fake_logits = fake_fxn(coords)

            else:
                gen_img = G(noise)
                fake_logits = D(gen_img)

            G_loss = G_loss_tracker(fake_logits)
            G_optim.zero_grad(set_to_none=True)
            G_loss.backward()
            G_optim.step()
            G_loss = G_loss.item()
            if np.isnan(G_loss):
                print('NaN G loss')
                pdb.set_trace()

            del fake_logits
            torch.cuda.empty_cache()

        else:
            with torch.no_grad():
                if args["network"]['type'].startswith('inr'):
                    gen_inr = G(noise)
                else:
                    gen_img = G(noise)

        if args["network"]['type'].startswith('inr'):
            fake_fxn = D(gen_inr.detach())
            true_fxn = D(true_inr)
            if dl_args['sample type'] == 'grid':
                fake_fxn.toggle_grid_mode(True)
                true_fxn.toggle_grid_mode(True)
            fake_logits = fake_fxn(coords)
            true_logits = true_fxn(coords)
            D_loss = D_loss_tracker(fake_logits, true_logits)
        else:
            true_img = true_inr.produce_images(*dl_args['image shape'])
            fake_logits = D(gen_img.detach())
            true_logits = D(true_img)
            gp = GP_tracker(true_img, gen_img, D)
            D_loss = D_loss_tracker(fake_logits, true_logits) + gp * 10
        
        D_optim.zero_grad(set_to_none=True)
        D_loss.backward()
        D_optim.step()
        if torch.isnan(D_loss):
            print('NaN D loss')
            pdb.set_trace()

        if global_step % 20 == 0:
            print("G:", np.round(G_loss, decimals=3), "; D:", np.round(D_loss.item(), decimals=3), flush=True)


        if global_step % 100 == 0:
            torch.save(G.state_dict(), osp.join(paths["weights dir"], "G.pth"))
            torch.save(D.state_dict(), osp.join(paths["weights dir"], "D.pth"))
            G_loss_tracker.plot_running_average(path=paths["job output dir"]+"/plots/G.png")
            D_loss_tracker.plot_running_average(path=paths["job output dir"]+"/plots/D.png")

            if args["network"]['type'].startswith('inr'):
                true = true_inr.produce_images(*dl_args['image shape'])[0]
                coords = true_inr.generate_sample_points(dims=dl_args['initial grid shape'], method='grid')
                gen_inr.toggle_grid_mode(True)
                fake = gen_inr(coords)[0]
                coll_img = torch.cat((rescale_float(true.permute(1,2,0)),
                    rescale_float(fake.reshape(*dl_args['image shape'],3))), dim=1)
                coll_img = F.interpolate(coll_img.unsqueeze(0).unsqueeze(0), size=(150,300,3)).squeeze()
                plt.imsave(paths["job output dir"]+f"/imgs/{global_step}.png", coll_img.detach().cpu().numpy())
            else:
                true = rescale_float(true_img[0].permute(1,2,0))
                fake = rescale_float(gen_img[0].permute(1,2,0))
                coll_img = torch.cat((true,fake), dim=1)
                coll_img = F.interpolate(coll_img.unsqueeze(0).unsqueeze(0), size=(150,300,3)).squeeze()
                plt.imsave(paths["job output dir"]+f"/imgs/{global_step}.png", coll_img.detach().cpu().numpy())
            
        if global_step >= args["optimizer"]["max steps"]:
            break

    torch.save(G.state_dict(), osp.join(paths["weights dir"], "G.pth"))
    torch.save(D.state_dict(), osp.join(paths["weights dir"], "D.pth"))


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
