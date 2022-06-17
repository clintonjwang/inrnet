"""
Entrypoint for training
"""
import sys
import torch, wandb
import numpy as np
from functools import partial

from inrnet import args as args_module
# from inrnet.experiments.diffusion import train_diffusion_model
# from inrnet.experiments.depth import train_depth_model
from inrnet.experiments.sdf import train_nerf_to_sdf
from inrnet.experiments.classify import train_classifier
from inrnet.experiments.segment import train_segmenter
# from inrnet.experiments.generate import train_generator
# from inrnet.experiments.warp import train_warp

def main():
    args = args_module.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
    if args["random seed"] >= 0:
        np.random.seed(args["random seed"])
        torch.manual_seed(args["random seed"])
    method_dict = {
        # 'diffusion': train_diffusion_model,
        'classify': train_classifier,
        'sdf': train_nerf_to_sdf,
        # 'depth': train_depth_model,
        'segment': train_segmenter,
        # 'generate': train_generator,
        # 'warp': train_warp,
    }
    method = method_dict[args["network"]["task"]]
    if args['sweep_id'] is not None:
        wandb.agent(args['sweep_id'], function=partial(method, args=args), count=1, project='inrnet')
    else:
        method(args=args)

if __name__ == "__main__":
    main()
