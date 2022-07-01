"""Entrypoint for inference"""
import sys
import torch
import numpy as np

from inrnet.utils import args as args_module
from inrnet.experiments.classify import test_inr_classifier
from inrnet.experiments.segment import test_inr_segmenter
from inrnet.experiments.generate import test_inr_generator
# from inrnet.experiments.sdf import test_nerf_to_sdf

def main():
    args = args_module.parse_args(sys.argv[1:])
    if not torch.cuda.is_available():
        raise ValueError("cuda is not available on this device")
    torch.backends.cudnn.benchmark = True
    if args["random seed"] >= 0:
        np.random.seed(args["random seed"])
        torch.manual_seed(args["random seed"])

    method_dict = {
        'classify': test_inr_classifier,
        # 'sdf': train_nerf_to_sdf,
        'segment': test_inr_segmenter,
        'generate': test_inr_generator,
    }
    test_inr_segmenter(method_dict[args["network"]["task"]])

if __name__ == "__main__":
    main()
