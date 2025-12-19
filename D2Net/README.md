# D2-Net: Dual Disentanglement Network for Brain Tumor Segmentation with Missing Modalities

[[Paper]](https://ieeexplore.ieee.org/document/9775681) [[Code]](https://github.com/CityU-AIM-Group/D2Net) IEEE TMI 2022
![D2Net overview](/D2Net/fig/D2Net.png)

✅ Tested at commit: 8359e49

## Requirements
This code was tested using:
```
python==3.11.5
torch==2.7.1+cu128
```
Detailed versioning of every package can be found in the requirements.txt file

This code currently runs only on machines with CUDA support. GPU access is required for all processing steps.
## How to run
Before running any code, ensure you have correctly downloaded the BraTS 2023 Challenge dataset, specifically the subset for [Glioma Segmentation](https://www.synapse.org/Synapse:syn51156910/wiki/622351)\
\
Clone this repository, create a python env for the project and activate it. Then install all the dependencies with pip.
```
cd D2Net
python -m venv d2net_venv
source d2net_venv/bin/activate
pip install -r requirements.txt
```
If you want to track the training using wandb, setup the wandb library following [this guide](https://docs.wandb.ai/models/quickstart).

## Preprocessing
Run this code for preprocessing
```
cd code/
python preprocess.py
  --input-path <INPUT_PATH> \                  # Directory with the original BraTS dataset
  --output-path <OUTPUT_PATH> \                # Directory for the preprocessed dataset
  --interactive
```

## Training
To train the model run:
```
python train.py\
  --datapath <INPUT_PATH> \                    # Directory with the preprocessed dataset
  --num-epochs 130 \                           # Number of epoch
  --checkpoint-path <CHECKPOINT_PATH> \        # Directory for saving the checkpoints
  --wandb-project-name D2Net \                 # Optional, allows for wandb tracking
  --num-workers <NUM_WORKERS> \                # Number of workers of the dataloaders
  --batch-size <BATCH SIZE> \                  # Batch size. Start with 4
```
## Testing
After training, to test the model run:
```
python test.py \
  --datapath <INPUT_PATH> \                    # Directory with the preprocessed dataset
  --savepath <OUTPUT_PATH> \                   # Directory for saving results
  --resume <RESUME_PATH> \                     # Path to the checkpoints 
  --batch_size 2 \                             # Batch size
```
