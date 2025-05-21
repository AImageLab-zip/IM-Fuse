# mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation

[[Paper]](https://arxiv.org/abs/2206.02425) [[Code]](https://github.com/YaoZhang93/mmFormer) MICCAI 2022 

<p align="center">
  <img src="figs/image.png">
</p>

## How to run
Run mmFormer using the same python environment and data preprocessing of IM-Fuse.
```
cd mmFormer
source imfuse_venv/bin/activate
```

## Training
Run the training script `train.py` with the following arguments:
```
python train.py \
  --datapath <PATH>/BRATS2023_Training_npy \   
  --num_epochs 1000 \                          
  --dataname BRATS2023 \                       
  --savepath <OUTPUT_PATH> \                   
  --batch_size 3                               
```

## Test
Run the test script `test.py` setting the data path and the path to the checkpoint in the script.
