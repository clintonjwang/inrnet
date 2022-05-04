import os, pdb, torch
osp = os.path
import torch
nn = torch.nn
F = nn.functional

from data import dataloader
import inn

TMP_DIR = osp.expanduser("~/code/diffcoord/temp")
rescale_float = mtr.ScaleIntensity()

def finetune_segmenter(args):
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)

    global_step = 0
    InrNet = getSegNet(args).train()
    optimizer = torch.optim.Adam(InrNet.parameters(), lr=args["optimizer"]["learning rate"])
    H,W = dl_args["image shape"]
    loss_tracker = util.MetricTracker("loss", function=losses.L1_dist)
    for img_inr, seg in data_loader:
        global_step += 1
        xyz[0,:,0] /= W/2
        xyz[0,:,1] /= H/2
        xyz[0,:,:2] -= 1
        xyz = xyz.half()
        img_inr = to_black_box(img_inr)
        Seg_inr = InrNet(img_inr)
        seg_pred = Seg_inr(xyz[0,:,:2])
        loss = loss_tracker(z_pred, xyz[0,:,-1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if global_step % 10 == 0:
            print(loss.item(),flush=True)
            del loss, z_pred
            torch.cuda.empty_cache()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
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

            torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "best.pth"))

            # loss_tracker.plot_running_average(root=paths["job output dir"]+"/plots")
            # util.save_examples(global_step, paths["job output dir"]+"/imgs",
            #     example_outputs["orig_imgs"][:2], example_outputs["fake_img"][:2],
            #     example_outputs["recon_img"][:2], transforms=transforms)

        # if attr_tracker.is_at_min("train"):
        #     torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "best.pth"))

        if global_step > args["optimizer"]["max steps"]:
            break

    torch.save(InrNet.state_dict(), osp.join(paths["weights dir"], "final.pth"))
