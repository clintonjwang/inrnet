import os, pdb, torch
osp = os.path
import torch
nn = torch.nn
F = nn.functional

import monai.transforms as mtr
from inrnet.data import dataloader
from inrnet import inn

TMP_DIR = osp.expanduser("~/code/diffcoord/temp")
rescale_float = mtr.ScaleIntensity()

def train_diffusion_model(args):
    paths=args["paths"]
    # loss_args=args["loss"]
    # model = model_module.get_model(paths, args).train()
    # optimizer = optim.get_optimizer(model, optim_args)
    # gradscaler = torch.cuda.amp.GradScaler()

    inr_size = 198915
    i2i = baseline.DiffusionINR2INR(inr_size=inr_size, C=128).cuda()
    diffusion = ddpm.GaussianDiffusion(
        i2i, timesteps = 1000,   # number of steps
        loss_type = 'l1',    # L1 or L2
        inr_size = inr_size,
    ).cuda()
    optimizer = optim.get_optimizer(i2i, args["optimizer"])

    data_paths = util.glob2(TMP_DIR, "siren_*.pt")

    eval_interval = 500
    global_step = 0
    max_steps = args["optimizer"]["max steps"]
    # intervals = {"train":1, "val":eval_interval}
    N_samples = 2
    siren = img2d.Siren(H=400,W=400).eval().cuda()

    for path in util.cycle(data_paths):
        global_step += 1
        param_dicts = torch.load(path)
        keys = ['net.0.linear.weight', 'net.0.linear.bias', 'net.1.linear.weight', 'net.1.linear.bias', 'net.2.linear.weight', 'net.2.linear.bias', 'net.3.linear.weight', 'net.3.linear.bias', 'net.4.weight', 'net.4.bias']
        shape_by_key = {}
        inputs = []
        for k in keys:
            param_dicts[k] = torch.stack(param_dicts[k],0)
            shape_by_key[k] = param_dicts[k].shape[1:]
            inputs.append(param_dicts[k].flatten(start_dim=1))
        inputs = torch.cat(inputs, dim=1)

        optimizer.zero_grad()
        #with torch.cuda.amp.autocast():
        loss = diffusion(inputs.cuda())
        loss.backward()
        optimizer.step()

        if global_step % eval_interval == 0 or global_step >= max_steps:
            print(global_step, loss.item())
            sampled_params = diffusion.sample(batch_size=N_samples)

            ix = 0
            new_param_dict = {}
            for k in keys:
                cur_shape = shape_by_key[k]
                nelem = int(np.prod(cur_shape))
                new_param_dict[k] = sampled_params[:, ix:ix+nelem].reshape(-1, *cur_shape)
                ix+=nelem
            assert ix == sampled_params.size(1)

            for i in range(N_samples):
                single_param_dict = {k:v[i] for k,v in new_param_dict.items()}
                siren.load_state_dict(single_param_dict)
                rgb = rescale_float(siren.produce_image().detach().cpu().numpy())
                plt.imsave(paths["job output dir"]+f"/imgs/{global_step//100}_{i}.png", rgb)

        if global_step >= max_steps:
            break
