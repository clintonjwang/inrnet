# Deep Learning on Implicit Neural Datasets

[![Build Status](https://app.travis-ci.com/clintonjwang/inrnet.svg?token=VtxpFkfJv6myVXJHSmKW&branch=main)](https://app.travis-ci.com/clintonjwang/inrnet)

**INR-Net** is a principled deep learning framework for learning and inference directly with implicit neural representations (INRs) of any type without reverting to grid-based features or operations. INR-Nets evaluate INRs on a low discrepancy sequence, enabling quasi-Monte Carlo (QMC) integration throughout the network. We prove INR-Nets are universal approximators on a large class of maps between $L^2$ functions. Additionally, INR-Nets have convergent gradients under the empirical measure, enabling backpropagation. We design INR-Nets as a continuous generalization of discrete networks, enabling them to be initialized with pre-trained models. We demonstrate learning of INR-Nets on classification (INR$\to$label) and segmentation (INR$\to$INR) tasks.
