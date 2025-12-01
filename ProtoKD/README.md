# Prototype Knowledge Distillation for Medical Segmentation with Missing Modality
[[Paper]](https://arxiv.org/pdf/2303.09830) [[Code]](https://github.com/SakurajimaMaiii/ProtoKD) ICASSP 2023
![ProtoKD overview](/ProtoKD/ProtoKD.png)

## Requirements
This code was tested using:
```
python==3.11.5
torch==2.7.1+cu128
```
## How to run
Before running any code, ensure you have correctly downloaded the BraTS 2023 Challenge dataset, specifically the subset for [Glioma Segmentation](https://www.synapse.org/Synapse:syn51156910/wiki/622351)\
\
Clone this repository, create a python env for the project and activate it. Then install all the dependencies with pip.
```
cd ProtoKD
python -m venv protokd_venv
source protokd_venv/bin/activate
pip install -r requirements.txt
```
If you want to track the training using wandb, setup the wandb library following [this guide](https://docs.wandb.ai/models/quickstart).
## Preprocessing
Run this code for preprocessing
```
cd code/
python preprocess.py
  --input-path <INPUT_PATH>                    # Directory with the original BraTS dataset
  --output-path <OUTPUT_PATH>                  # Directory for the preprocessed dataset
```
## Training
First, train the teacher model.
```
python pretrain.py
  --datapath <PATH>/BRATS2023_Training_npy \   # Directory containing BRATS2023 .npy files
  --num-epochs 229 \                           # Total number of training epochs
  --savepath <OUTPUT_PATH> \                   # Directory for saving checkpoints 
  --wandb_project_name ProtoKD \               # Add this option to enable wandb tracking
  --cache-dataset                              # Optional, loads the dataset on RAM
```
After completing the training of the teacher model, you can launch the second script.
You will need to run it separately for each desired combination of available and missing modalities.
This configuration is specified using the --modalities argument, which takes a 4-character string: use 0 for a missing modality and 1 for an available one, in the following order: t1n, t2w, t1c, t2f.

For example:
```--modalities 0101``` means t1n is missing, t2w is available, t1c is missing, and t2f is available.

python train.py
  --datapath <PATH>/BRATS2023_Training_npy \   # Directory containing BRATS2023 .npy files
  --num_epochs 229 \                           # Total number of training epochs
  --savepath <OUTPUT_PATH> \                   # Directory for saving checkpoints 
  --teacher-checkpoint <CHECKPOINT_PATH> \     # Path to the teacher's checkpoint
  --wandb_project_name ProtoKD \               # Add this option to enable wandb tracking
  --modalities <XXXX> \                        # String representing the missing and available modalities
  --cache-dataset                              # Optional, loads the dataset on RAM

