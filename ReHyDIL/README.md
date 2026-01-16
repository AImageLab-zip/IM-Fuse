# Hypergraph Tversky-Aware Domain Incremental Learning for Brain Tumor Segmentation with Missing Modalities

[[Paper]](https://papers.miccai.org/miccai-2025/paper/2774_paper.pdf) [[Code]](https://github.com/reeive/ReHyDIL) MICCAI 2025
![ReHyDIL overview](/ReHyDIL/pic/miccai25-rehydil.png)

✅ Tested at commit: 
??????? 

## Requirements
This code was tested using:
```
python==3.11.5
torch==2.9.1
```
Detailed versioning of every package can be found in the requirements.txt file

This code currently runs only on machines with CUDA support. GPU access is required for all processing steps.
## How to run
Before running any code, ensure you have correctly downloaded the BraTS 2023 Challenge dataset, specifically the subset for [Glioma Segmentation](https://www.synapse.org/Synapse:syn51156910/wiki/622351)\
\
Clone this repository, create a python env for the project and activate it. Then install all the dependencies with pip.
```
cd ReHyDIL
python3.11 -m venv rehydil_venv
source rehydil_venv/bin/activate
pip install -r requirements.txt
```
If you want to track the training using wandb, setup the wandb library following [this guide](https://docs.wandb.ai/models/quickstart).
## Preprocessing
First, run the preprocessing pipeline:
```
python preprocess.py\
  --datapath <INPUT_PATH> \                    # Directory with the original dataset
  --outputpath <OUTPUT_path>                   # Directory for the new, preprocessed, dataset
  --num-workers <NUM_WORKERS>                  # Optional, number of workers for parallel preprocessing
  --interactive                                # Optional, allows for interactive directory deletion
```

## Training
The model works by training the model in four consecutive stages.
### Stage 1
```
python train.py\
  --datapath <INPUT_PATH> \                    # Directory with the preprocessed dataset. USE THE SAME AS BEFORE
  --current-stage t1n                          # Fist stage
  --num-epochs 50 \                            # Number of epoch per stage
  --checkpoint-path <CHECKPOINT_PATH> \        # Directory for saving the checkpoints
  --wandb-project-name PROJECT NAME \          # Optional, allows for wandb tracking
  --num-workers <NUM_WORKERS> \                # Number of workers of the dataloaders
  --batch-size <BATCH SIZE> \                  # Batch size. Start with 16
```
### Stage 2
```
python train.py \
  --datapath <INPUT_PATH> \                    # Directory with the preprocessed dataset. USE THE SAME AS BEFORE
  --current-stage t2w                          # Second stage
  --num-epochs 50 \                            # Number of epoch per stage
  --checkpoint-path <CHECKPOINT_PATH> \        # Directory for saving the checkpoints
  --wandb-project-name PROJECT NAME \          # Optional, allows for wandb tracking
  --num-workers <NUM_WORKERS> \                # Number of workers of the dataloaders
  --batch-size <BATCH SIZE> \                  # Batch size. Start with 16
```
### Stage 3
```
python train.py \
  --datapath <INPUT_PATH> \                    # Directory with the preprocessed dataset. USE THE SAME AS BEFORE
  --current-stage t2f                          # Third stage
  --num-epochs 80 \                            # Number of epoch per stage
  --checkpoint-path <CHECKPOINT_PATH> \        # Directory for saving the checkpoints
  --wandb-project-name PROJECT NAME \          # Optional, allows for wandb tracking
  --num-workers <NUM_WORKERS> \                # Number of workers of the dataloaders
  --batch-size <BATCH SIZE> \                  # Batch size. Start with 16
```
### Stage 4
```
python train.py \
  --datapath <INPUT_PATH> \                    # Directory with the preprocessed dataset. USE THE SAME AS BEFORE
  --current-stage t1c                          # Fourth stage
  --num-epochs 80 \                            # Number of epoch per stage
  --checkpoint-path <CHECKPOINT_PATH> \        # Directory for saving the checkpoints
  --wandb-project-name PROJECT NAME \          # Optional, allows for wandb tracking
  --num-workers <NUM_WORKERS> \                # Number of workers of the dataloaders
  --batch-size <BATCH SIZE> \                  # Batch size. Start with 16
```
## Testing
After training, to test the model run:
```
python test.py \
  --datapath <INPUT_PATH> \                    # Directory with the preprocessed dataset
  --savepath <OUTPUT_PATH> \                   # Directory for saving results
  --resume <RESUME_PATH> \                     # Path to the checkpoints 
```