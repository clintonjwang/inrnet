"""Entrypoint for inference"""
import torch
import numpy as np

from inrnet import args as args_module
from inrnet.experiments.classify import test_inr_classifier
from inrnet.experiments.segment import test_inr_segmenter
from inrnet.experiments.generate import test_inr_generator

def main():
    args = args_module.parse_args()
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
    if args["random seed"] >= 0:
        np.random.seed(args["random seed"])
        torch.manual_seed(args["random seed"])

    if args["network"]["task"] == "segment":
        test_inr_segmenter(args)
    elif args["network"]["task"] == "classify":
        test_inr_classifier(args)
    elif args["network"]["task"] == "generate":
        test_inr_generator(args)

if __name__ == "__main__":
    main()
