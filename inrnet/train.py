import torch
import numpy as np

import args as args_module
from experiments.diffusion import train_diffusion_model
from experiments.depth import train_depth_model
from experiments.classify import train_classifier
from experiments.segment import train_segmenter
from experiments.generate import train_generator

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
        train_classifier(args)
    elif args["network"]["task"] == "depth":
        train_depth_model(args)
    elif args["network"]["task"] == "cyclegan":
        train_cyclegan(args)
    elif args["network"]["task"] == "segment":
        train_segmenter(args)
    elif args["network"]["task"] == "generate":
        train_generator(args)

if __name__ == "__main__":
    args = args_module.parse_args()
    main(args)
