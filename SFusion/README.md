# SFusion: Self-attention based N-to-One Fusion Block
[[Paper]](https://arxiv.org/abs/2208.12776) [[Code]](https://github.com/scut-cszcl/SFusion) MICCAI 2023
<p align="center">
  <img src="fig/image.png">
</p>


## How to run
Run SFusion using the same python environment and data preprocessing of IM-Fuse.
```
cd SFusion
source imfuse_venv/bin/activate
```

## Training
Run the training script `train_sfusion.py` with the following arguments:
```
python train_sfusion.py \
  --datapath <PATH>/BRATS2023_Training_npy \   
  --num_epochs 200 \                           
  --dataname BRATS2023 \                       
  --savepath <OUTPUT_PATH> \                    
  --batch_size 1                               
  --feature_maps 16                            
```

## Test
Run the test script `test_sfusion.py` setting the data path and the path to the checkpoint in the script.
