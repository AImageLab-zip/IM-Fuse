# Reversible Feature Learning for Brain Tumor Segmentation with Incomplete Modalities
The code of our rfl feature learning method
# Installation
```python
pip install -r requirements.txt
```
# Data preparation
* Download the preprocessed BraTS2018 dataset from [RFNet](https://drive.google.com/drive/folders/1AwLwGgEBQwesIDTlWpubbwqxxd8brt5A)
* For BraTS2021, download the train dataset from [here](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1), change the path in `preprocess.py`, and run:
```python
python process.py
```
# Training
Change the paths and hyperparameters in `train.py`, then run:
```python
python train.py --batch_size=x --datapath xxx --savepath xxx --num_epochs xxx --dataname BRATS20xx
```
# Test
The trained model should be located in `rfl/output`, then run:
```python
python train.py --batch_size=x --datapath xxx --savepath xxx --num_epochs 0 --dataname BRATS20xx --resume xxx
```
The resume is the path of trained model
