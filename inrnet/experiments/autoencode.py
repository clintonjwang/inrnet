
import wandb
from inrnet import inn
from inrnet.data import dataloader
from inrnet.utils import args as args_module, jobs as job_mgmt, util


def pretrain_inrnet(args: dict) -> None:
    if not args['no_wandb']:
        wandb.init(project="inrnet", job_type="train", name=args["job_id"],
            config=wandb.helper.parse_config(args, exclude=['job_id']))
        args = args_module.get_wandb_config()
    paths = args["paths"]
    dl_args = args["data loading"]
    data_loader = dataloader.get_inr_dataloader(dl_args)
    val_data_loader = dataloader.get_val_inr_dataloader(dl_args)
    global_step = 0