# IM-Fuse: A Mamba-based Fusion Block for Brain Tumor Segmentation with Incomplete Modalities
This is the implementation for the paper:

IM-Fuse: A Mamba-based Fusion Block for Brain tumor Segmentation with Incomplete Modalities

Submitted to MICCAI 2025

## Abstract

Brain tumor segmentation is a crucial task in medical imaging that involves the integrated modeling of four distinct imaging modalities to identify tumor regions accurately. Unfortunately, in real-life scenarios, the full availability of such four modalities is often violated due to scanning cost, time, and patient condition. Consequently, several deep learning models have been developed to address the challenge of brain tumor segmentation under conditions of missing imaging modalities. However, the majority of these models have been evaluated using the 2018 version of the BraTS dataset, which comprises only $285$ volumes. 
In this study, we reproduce and extensively analyze the most relevant models using BraTS2023, which includes $1,250$ volumes, thereby providing a more comprehensive and reliable comparison of their performance. Furthermore, we propose and evaluate the adoption of Mamba as an alternative fusion mechanism for brain tumor segmentation in the presence of missing modalities. Experimental results demonstrate that transformer-based architectures achieve leading performance on BraTS2023, outperforming purely convolutional models that were instead superior in BraTS2018. Meanwhile, the proposed Mamba-based architecture exhibits promising performance in comparison to state-of-the-art models, competing and even outperforming transformers.

In the following figure (a) is an overview of our framework IM-Fuse (Incomplete Modality Fusion), (b) represents our Mamba Fusion Block (MFB) where learnable tokens are concatenated, and (c) depicts its interleaved version (Interleaved-MFB or I-MFB) where modality tokens and learnable parameters are alternately arranged.

![image](https://github.com/AImageLab-zip/IM-Fuse/blob/main/figs/IM-Fuse-overview.png)
