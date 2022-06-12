"""
Entrypoint for pretraining
"""
import sys
import torch, wandb
import numpy as np
from functools import partial

from inrnet import args as args_module
from inrnet.experiments.autoencode import pretrain_inrnet

def main():
    args = args_module.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
    if args["random seed"] >= 0:
        np.random.seed(args["random seed"])
        torch.manual_seed(args["random seed"])
    if args['sweep_id'] is not None:
        wandb.agent(args['sweep_id'], function=partial(pretrain_inrnet, args=args), count=1, project='inrnet')
    else:
        pretrain_inrnet(args=args)

if __name__ == "__main__":
    main()
