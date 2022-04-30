import torch
import numpy as np

import args as args_module
from experiments.diffusion import train_diffusion_model
from experiments.depth import train_depth_model
from experiments.classify import finetune_classifier
from experiments.cyclegan import train_cyclegan

def main(args):
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
    if args["random seed"] >= 0:
        np.random.seed(args["random seed"])
        torch.manual_seed(args["random seed"])

    if args["network"]["task"] == "diffusion":
        train_diffusion_model(args)
    elif args["network"]["task"] == "classify":
        finetune_classifier(args)
    elif args["network"]["task"] == "depth":
        train_depth_model(args)
    elif args["network"]["task"] == "cyclegan":
        train_cyclegan(args)

if __name__ == "__main__":
    args = args_module.parse_args()
    main(args)
