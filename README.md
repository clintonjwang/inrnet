# Deep Learning on Implicit Neural Datasets

[![main-workflow](https://github.com/clintonjwang/inrnet/actions/workflows/config.yml/badge.svg)](https://github.com/clintonjwang/inrnet/actions/workflows/config.yml)
[![codecov](https://codecov.io/gh/clintonjwang/inrnet/branch/main/graph/badge.svg?token=IEFLBO4XHF)](https://codecov.io/gh/clintonjwang/inrnet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/clintonjwang/inrnet/blob/main/LICENSE)

![INR-Nets learn directly from datasets of implicit neural representations](https://github.com/clintonjwang/inrnet/blob/main/teaser.png?raw=true)

**INR-Net** is a principled deep learning framework for learning and inference directly with implicit neural representations (INRs) of any type without reverting to grid-based features or operations. INR-Nets evaluate INRs on a low discrepancy sequence, enabling quasi-Monte Carlo (QMC) integration throughout the network. We prove INR-Nets are universal approximators on a large class of maps between $L^2$ functions. Additionally, INR-Nets have convergent gradients under the empirical measure, enabling backpropagation. We design INR-Nets as a continuous generalization of discrete networks, enabling them to be initialized with pre-trained models. We demonstrate learning of INR-Nets on classification (INR&rarr;label) and segmentation (INR&rarr;INR) tasks.


## Code Usage

TBD

## Requirements

CUDA is required. The code is not written to run on CPUs.

## Citation

**[Deep Learning on Implicit Neural Datasets](https://arxiv.org/abs/2206.01178)**<br>
[Clinton J. Wang](https://clintonjwang.github.io/) and [Polina Golland](https://people.csail.mit.edu/polina/)<br>
arXiv preprint 2022

If you find this work useful please use the following citation:
```
@misc{wang2022deep,
      title={Deep Learning on Implicit Neural Datasets}, 
      author={Clinton J. Wang and Polina Golland},
      year={2022},
      eprint={2206.01178},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Acknowledgements

Thanks to [Daniel Moyer](https://dcmoyer.github.io/) for his many helpful suggestions.
