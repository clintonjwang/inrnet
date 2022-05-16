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

rescale_float = mtr.ScaleIntensity()

n_ch = 1
def load_pretrained_model(args):
    net_args = args["network"]
    pretrained = net_args['pretrained']
    if isinstance(pretrained, str):
        raise NotImplementedError('cant pretrain')
        base = load_model_from_job(pretrained)
    else:
        if net_args["type"] == "wgan":
            G,D = inrnet.models.wgan.wgan()
        elif net_args["type"] == "inr-wgan":
            G,D = inrnet.inn.nets.wgan.translate_wgan_model()
        elif net_args["type"] == "inr-4":
            G,D = inrnet.inn.nets.wgan.Gan4(reshape=args['data loading']['initial grid shape'])
        elif net_args["type"] == "cnn-4":
            G,D = inrnet.models.wgan.Gan4()
        else:
            raise NotImplementedError('bad net type')
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
    if args["network"]['type'].startswith('inr'):
        GP_tracker = util.MetricTracker("GP", function=losses.gradient_penalty_inr)
    else:
        GP_tracker = util.MetricTracker("GP", function=losses.gradient_penalty)

    bsz = dl_args['batch size']
    D_step = 1
    # G_step = 1

    G,D = load_pretrained_model(args)
    G_optim = torch.optim.AdamW(G.parameters(), lr=args["optimizer"]["G learning rate"], betas=(.5,.999))
    D_optim = torch.optim.AdamW(D.parameters(), lr=args["optimizer"]["D learning rate"], betas=(.5,.999))
    for true_inr in data_loader:
        global_step += 1
        noise = torch.randn(dl_args["batch size"], 64, device='cuda')
        if global_step % D_step == 0:
            if args["network"]['type'].startswith('inr'):
                gen_inr = G(noise)
                fake_fxn = D(gen_inr)
                if dl_args['sample type'] == 'qmc':
                    coords = fake_fxn.generate_sample_points(sample_size=dl_args["initial sample points"], method='qmc')
                else:
                    fake_fxn.change_sample_mode('grid')
                    coords = fake_fxn.generate_sample_points(dims=dl_args['initial grid shape'], method='grid')
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
                    if dl_args['sample type'] == 'qmc':
                        coords = gen_inr.generate_sample_points(sample_size=dl_args["initial sample points"], method='qmc')
                    else:
                        gen_inr.change_sample_mode('grid')
                        coords = gen_inr.generate_sample_points(dims=dl_args['initial grid shape'], method='grid')
                else:
                    gen_img = G(noise)

        # if global_step % G_step == 0:
        if args["network"]['type'].startswith('inr'):
            fake_fxn = D(gen_inr.detach())
            true_fxn = D(true_inr)
            if dl_args['sample type'] == 'grid':
                fake_fxn.change_sample_mode('grid')
                true_fxn.change_sample_mode('grid')
                true_coords = true_inr.generate_sample_points(dims=dl_args['image shape'], method='grid')
            elif dl_args['sample type'] == 'qmc':
                true_coords = true_inr.generate_sample_points(sample_size=dl_args["sample points"], method='qmc')
            fake_logits = fake_fxn(coords)
            true_logits = true_fxn(true_coords)
            gp = GP_tracker(true_coords, true_inr, gen_inr, D)
            D_loss = D_loss_tracker(fake_logits, true_logits) + gp * 10
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

        if global_step % 50 == 0:
            torch.save(G.state_dict(), osp.join(paths["weights dir"], "G.pth"))
            torch.save(D.state_dict(), osp.join(paths["weights dir"], "D.pth"))
            G_loss_tracker.plot_running_average(path=paths["job output dir"]+"/plots/G.png")
            D_loss_tracker.plot_running_average(path=paths["job output dir"]+"/plots/D.png")
            GP_tracker.plot_running_average(path=paths["job output dir"]+"/plots/gp.png")

            if args["network"]['type'].startswith('inr'):
                true = true_inr.produce_images(*dl_args['image shape'])[0]
                coords = true_inr.generate_sample_points(dims=dl_args['initial grid shape'], method='grid')
                gen_inr.change_sample_mode('grid')
                fake = gen_inr(coords)[:1]
                fake = util.realign_values(fake, inr=gen_inr)
                # fake = fake.clamp(min=-1,max=1)
                coll_img = torch.cat((rescale_float(true.permute(1,2,0)), rescale_float(fake[0].reshape(*dl_args['image shape'],n_ch))), dim=1)
                coll_img = F.interpolate(coll_img.unsqueeze(0).unsqueeze(0), size=(150,300,n_ch)).squeeze()
                plt.imsave(paths["job output dir"]+f"/imgs/{global_step}.png", coll_img.detach().cpu().numpy(), cmap='gray')
            else:
                true = rescale_float(true_img[0].permute(1,2,0))
                fake = rescale_float(gen_img[0].permute(1,2,0))
                coll_img = torch.cat((true,fake), dim=1)
                coll_img = F.interpolate(coll_img.unsqueeze(0).unsqueeze(0), size=(150,300,n_ch)).squeeze()
                plt.imsave(paths["job output dir"]+f"/imgs/{global_step}.png", coll_img.detach().cpu().numpy(), cmap='gray')
            
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
    G = load_model_from_job(origin)[0].cuda().eval()
    orig_args = job_mgmt.get_job_args(origin)
    bsz = dl_args['batch size']

    for ix in range(0, num_samples, bsz):
        noise = torch.randn(bsz, 64, device='cuda')
        noise = torch.where((noise>1) | (noise<-1), torch.randn_like(noise), noise)
        if args["network"]['type'].startswith('inr'):
            gen_inr = G(noise).eval()
            gen_inr.change_sample_mode('grid')
            coords = gen_inr.generate_sample_points(dims=dl_args['initial grid shape'], method='grid')
            gen_img = gen_inr(coords)
            gen_img = util.realign_values(gen_img, inr=gen_inr)
        else:
            gen_img = G(noise)

        for i in range(bsz):
            if args["network"]['type'].startswith('inr'):
                gen_img = rescale_float(gen_img[i].reshape(*dl_args['image shape'],n_ch))
            else:
                gen_img = rescale_float(gen_img[i].permute(1,2,0))
                
        gen_img = F.interpolate(gen_img.unsqueeze(0).unsqueeze(0), size=(150,150,n_ch)).squeeze()
        plt.imsave(paths["job output dir"]+f"/imgs/{ix+i}.png", gen_img.detach().cpu().numpy(), cmap='gray')
