# IM-Fuse: A Mamba-based Fusion Block for Brain Tumor Segmentation with Incomplete Modalities
[[Our Paper]]() MICCAI 2025

![IMFuse overview](/figs/IM-Fuse-overview.png)

## How to run
Clone this repository, create a python env for the project and activate it. Then install all the dependencies with pip.
```
git clone git@github.com:AImageLab-zip/IM-Fuse.git
cd IMFuse
python -m venv imfuse_venv
source imfuse_venv/bin/activate
pip install -r requirements.txt
```
## Preprocess data
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
