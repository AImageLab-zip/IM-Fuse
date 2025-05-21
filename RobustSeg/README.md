# Robust Multimodal Brain Tumor Segmentation via Feature Disentanglement and Gated Fusion
[[Paper]](https://arxiv.org/abs/2002.09708) [[Code]](https://github.com/cchen-cc/Robust-Mseg) MICCAI 2019
<p align="center">
  <img src="fig/framework.png">
</p>

## How to run
Run Robust-Mseg using the same python environment and data preprocessing of IM-Fuse.
```
cd RobustSeg
source imfuse_venv/bin/activate
```

## Training
Run the training script `train_robustseg.py` with the following arguments:
```
python train_robustseg.py \
  --datapath <PATH>/BRATS2023_Training_npy \   
  --num_epochs 300 \                           
  --dataname BRATS2023 \                       
  --savepath <OUTPUT_PATH> \                   
  --batch_size 1                               
```

## Test
Run the test script `test_robustseg.py` setting the data path and the path to the checkpoint in the script.