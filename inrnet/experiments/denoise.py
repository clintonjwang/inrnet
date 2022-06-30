import wandb
from inrnet import inn
from inrnet.data import dataloader
from inrnet.inn.nets.transformer import NerfTransformer
import torch

from inrnet.utils import args as args_module, jobs as job_mgmt, util
nn = torch.nn

def get_denoising_model(args: dict):
    if args["network"]['type'].startswith('inr'):
        model = NerfTransformer(args["network"])
    else:
        raise NotImplementedError
    return model

def train_denoising_inrnet(args: dict) -> None:
    """
    Input: NeRFs trained on a small set of views.
    Output: NeRF which produces good views from unseen angles.
    """
    if not args['no_wandb']:
        wandb.init(project="inrnet", job_type="train", name=args["job_id"],
            config=wandb.helper.parse_config(args, exclude=['job_id']))
        args = args_module.get_wandb_config()
    paths = args["paths"]
    dl_args = args["data loading"]
    scene_dataloader = dataloader.get_scene_dataloader(dl_args)
    global_step = 0

    model = get_denoising_model(args).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    for scene in scene_dataloader:
        optimizer.zero_grad()
        nerf = scene['NeRF']
        poses_test = scene['held-out poses']
        gt_views_test = scene['held-out GT views']
        denoised_point_cloud = model(nerf)
        pred_views_test = util.project_point_cloud(denoised_point_cloud, poses=poses_test)
        loss = nn.MSELoss(pred_views_test, gt_views_test)
        loss.backward()
        optimizer.step()
        global_step += 1
        if global_step % args["logging"]["log_interval"] == 0:
            wandb.log({"loss": loss.item()})
        if global_step % args["logging"]["save_interval"] == 0:
            job_mgmt.save_checkpoint(model, optimizer, global_step, paths["checkpoint_dir"])
            