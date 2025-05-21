IM-Fuse: A Mamba-based Fusion Block for Brain Tumor Segmentation with Incomplete Modalities
<figure>
 <img style="float: left" src="figs/IM-Fuse-complete2.pdf" alt="Side view" width="100%">
 <figcaption><em>Overview of our framework IM-Fuse (Incomplete Modality Fusion), (b) represents our Mamba Fusion Block (MFB) where learnable tokens are concatenated, and (c) depicts its interleaved version (Interleaved-MFB or I-MFB) where modality tokens and learnable parameters are alternately arranged.</em></figcaption>
</figure>

## Introduction
This repository contains the material from the paper "IM-Fuse: A Mamba-based Fusion Block for Brain tumor Segmentation with Incomplete Modalities". It includes the implementation and scripts necessary to reproduce our BRATS2023 results using the IM‑Fuse model, alongside the implementations of leading prior methods: [Missing as Masking: Arbitrary Cross-modal Feature Reconstruction for Incomplete Multimodal Brain Tumor Segmentation](https://papers.miccai.org/miccai-2024/paper/0067_paper.pdf), [M3AE: Multimodal Representation Learning for Brain Tumor Segmentation with Missing Modalities](https://github.com/ccarliu/m3ae), [Multi-modal Learning with Missing Modality via Shared-Specific Feature Modelling](https://github.com/billhhh/ShaSpec), [SFusion: Self-attention based N-to-One Multimodal Fusion Block](https://github.com/scut-cszcl/SFusion), [mmformer: Multimodal medical transformer for incomplete multimodal learning of brain tumor segmentation](https://github.com/YaoZhang93/mmFormer), [Hetero-Modal Variational Encoder-Decoder for Joint Modality Completion and Segmentation](https://github.com/ReubenDo/U-HVED), [Robust Multimodal Brain Tumor Segmentation via Feature Disentanglement and Gated Fusion](https://github.com/cchen-cc/Robust-Mseg)

## Citing our work
[BibText](https://federicobolelli.it/pub_files/2025miccai_imfuse.html)

## IM-Fuse: A Mamba-based Fusion Block for Brain Tumor Segmentation with Incomplete Modalities
Brain tumor segmentation is a crucial task in medical imaging that involves the integrated modeling of four distinct imaging modalities to identify tumor regions accurately. Unfortunately, in real-life scenarios, the full availability of such four modalities is often violated due to scanning cost, time, and patient condition. Consequently, several deep learning models have been developed to address the challenge of brain tumor segmentation under conditions of missing imaging modalities. However, the majority of these models have been evaluated using the 2018 version of the BraTS dataset, which comprises only $285$ volumes. 
In this study, we reproduce and extensively analyze the most relevant models using BraTS2023, which includes $1,250$ volumes, thereby providing a more comprehensive and reliable comparison of their performance. Furthermore, we propose and evaluate the adoption of Mamba as an alternative fusion mechanism for brain tumor segmentation in the presence of missing modalities. Experimental results demonstrate that transformer-based architectures achieve leading performance on BraTS2023, outperforming purely convolutional models that were instead superior in BraTS2018. Meanwhile, the proposed Mamba-based architecture exhibits promising performance in comparison to state-of-the-art models, competing and even outperforming transformers.

## Dataset
Before running this project, you need to download the data from [BraTS 2023 Challenge](https://www.synapse.org/Synapse:syn51156910/wiki/) 

## IM-Fuse
### How to run
Clone this repository, create a python env for the project and activate it. Then install all the dependencies with pip
```
git clone git@github.com:AImageLab-zip/IM-Fuse.git
cd IMFuse
python -m venv imfuse_venv
source imfuse_venv/bin/activate
pip install -r requirements.txt
```
### Preprocess data
Set the data paths in `preprocess.py` and then run `python preprocess.py`.

### Training
Run the training script `train_poly.py` with the following arguments:
```
python train_poly.py \
  --datapath <PATH>/BRATS2023_Training_npy \   # Directory containing BRATS2023 .npy files
  --num_epochs 1000 \                          # Total number of training epochs
  --dataname BRATS2023 \                       # Dataset identifier
  --savepath <OUTPUT_PATH> \                   # Directory for saving checkpoints 
  --mamba_skip \                               # Using Mamba in the skip connections
  --interleaved_tokenization                   # Enable interleaved tokenization
```

### Test
Run the test script `test.py` with the following arguments:
```
python test.py \
  --datapath <PATH>/BRATS2023_Training_npy \   # Directory containing BRATS2023 .npy files
  --dataname BRATS2023 \                       # Dataset identifier
  --savepath <OUTPUT_PATH> \                   # Directory for saving results
  --resume <RESUME_PATH> \                     # Path to the checkpoints 
  --mamba_skip \                               # Using Mamba in the skip connections
  --batch_size 2 \                             # Batch size
  --interleaved_tokenization                   # Enable interleaved tokenization
```


## Missing as Masking
### How to run
Clone this repository, create a python env for the project and activate it. Then install all the dependencies with pip
```
cd MaM
python -m venv mam_venv
source mam_venv/bin/activate
pip install -r requirements.txt
```

### Preprocess data

### Training

### Test


## M3AE
### How to run
Clone this repository, create a python env for the project and activate it. Then install all the dependencies with pip
```
cd m3ae
python -m venv m3ae_venv
source m3ae_venv/bin/activate
pip install -r requirements.txt
```

### Pre-training
Perform a warm up with all modalities `pretrain.py` with the following arguments:
```
python pretrain.py \
--exp_name m3ae_pretrain \
--batch_size 2 \
--mdp 3 \
--dataset brats23 \
--mask_ratio 0.875 \
--lr 0.0003 \
```

### Training
Run the training script `train.py` with the following arguments:
```
python train.py \
--batch_size 2 \
--lr 0.0003 \
--model_type cnnnet \
--seed 999 \
--weight_kl 0.1 \
--feature_level 2 \
--epochs 300 \
--mdp 3 \
--wd 0.0001 \
--deep_supervised \
--patch_shape 128 \
--exp_name m3ae_train 
```

### Test
Run the test script `test.py` with the following arguments:
```
python test.py \
----checkpoint runs/m3ae_train/best.pth.tar \
--exp_name m3ae_train
```


## ShaSpec
### How to run
Clone this repository, create a python env for the project and activate it. Then install all the dependencies with pip
```
cd ShaSpec
python -m venv shaspec_venv
source shaspec_venv/bin/activate
pip install -r requirements.txt
```

### Warm up
Perform a warm up with all modalities `train_SS.py` with the following arguments:
```
python train_SS.py \
--snapshot_dir snapshots/BraTS23_ShaSpec_warmup \
--batch_size 1 \
--num_gpus 1 \
--num_steps 245000 \
--val_pred_every 875 \
--learning_rate 1e-2 \
--num_classes 3 \
--num_workers 4 \
--train_list BraTS23/BraTS23_train.csv \
--val_list BraTS23/BraTS23_val15splits.csv \
--test_list BraTS23/BraTS23_test15splits.csv \
--random_mirror \
--random_scale \
--weight_std \
--warm_up 
```

### Training
Run the training script `train_SS.py` with the following arguments:
```
python train_SS.py \
  --snapshot_dir /snaphots/BraTS23_ShaSpec_training \
  --batch_size 1 \
  --num_gpus 1 \
  --num_steps 406000 \
  --val_pred_every 875 \
  --learning_rate 1e-2 \
  --num_classes 3 \
  --num_workers 4 \
  --train_list BraTS23/BraTS23_train.csv \
  --val_list BraTS23/BraTS23_val15splits.csv \
  --test_list BraTS23/BraTS23_test15splits.csv \
  --random_mirror \
  --random_scale \
  --weight_std \
  --mode random \
  --reload_path ShaSpec/snapshots/BraTS23_ShaSpec_warmup/best.pth \
  --reload_from_checkpoint
```

### Test
Run the test script `eval.py` with the following arguments:
```
python eval.py \
  --num_classes 3 \
  --data_list BraTS23/BraTS23_test.csv \
  --weight_std \
  --restore_from /work/grana_neuro/missing_modalities/ShaSpec/snapshots/BraTS23_ShaSpec_training/final.pth \
  --mode <MODE>
```
where `<MODE>` indicates the missing modality scenario. In the command above, replace `<MODE>` with one of the following missing‑modality configurations to evaluate every scenario:
```
0
0,1
0,1,2
0,1,2,3
0,1,3
0,2
0,2,3
0,3
1
1,2
1,2,3
1,3
2
2,3
3
```

## mmFormer
### How to run
Run mmFormer using the same python env and data preprocessing of IM-Fuse.
```
cd mmFormer
source imfuse_venv/bin/activate
```

### Training
Run the training script `train.py` with the following arguments:
```
python train.py \
  --datapath <PATH>/BRATS2023_Training_npy \   # Directory containing BRATS2023 .npy files
  --num_epochs 1000 \                          # Total number of training epochs
  --dataname BRATS2023 \                       # Dataset identifier
  --savepath <OUTPUT_PATH> \                   # Directory for saving checkpoints 
  --batch_size 3                               # Batch size
```

### Test
Run the test script `test.py` setting the data path and the path to the checkpoint in the script.


## Robust-MSeg
### How to run
Run Robust-Mseg using the same python env and data preprocessing of IM-Fuse.
```
cd RobustSeg
source imfuse_venv/bin/activate
```

### Training
Run the training script `train_robustseg.py` with the following arguments:
```
python train_robustseg.py \
  --datapath <PATH>/BRATS2023_Training_npy \   # Directory containing BRATS2023 .npy files
  --num_epochs 300 \                           # Total number of training epochs
  --dataname BRATS2023 \                       # Dataset identifier
  --savepath <OUTPUT_PATH> \                   # Directory for saving checkpoints 
  --batch_size 1                               # Batch size
```

### Test
Run the test script `test_robustseg.py` setting the data path and the path to the checkpoint in the script.


## SFusion
### How to run
Run SFusion using the same python env and data preprocessing of IM-Fuse.
```
cd SFusion
source imfuse_venv/bin/activate
```

### Training
Run the training script `train_sfusion.py` with the following arguments:
```
python train_sfusion.py \
  --datapath <PATH>/BRATS2023_Training_npy \   # Directory containing BRATS2023 .npy files
  --num_epochs 200 \                           # Total number of training epochs
  --dataname BRATS2023 \                       # Dataset identifier
  --savepath <OUTPUT_PATH> \                   # Directory for saving checkpoints 
  --batch_size 1                               # Batch size
  --feature_maps 16                            # Hidden dimension of the model
```

### Test
Run the test script `test_sfusion.py` setting the data path and the path to the checkpoint in the script.


## U-HVED
### How to run
Run U-HVED using the same data preprocessing of IM-Fuse and creating the following python env:
```
cd UHVED
python -m venv uhved_venv
source uhved_venv/bin/activate
pip install -r requirements.txt
```

### Training
Run the training script `train_uhved.py` with the following arguments:
```
python train_uhved.py \
  --datapath <PATH>/BRATS2023_Training_npy \   # Directory containing BRATS2023 .npy files
  --num_epochs 400 \                           # Total number of training epochs
  --dataname BRATS2023 \                       # Dataset identifier
  --savepath <OUTPUT_PATH> \                   # Directory for saving checkpoints 
  --batch_size 1                               # Batch size
```

### Test
Run the test script `test_uhved.py` setting the data path and the path to the checkpoint in the script.


