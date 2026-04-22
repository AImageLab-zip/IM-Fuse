# UNT-MFI: Modality-adaptive Feature Interaction for Brain Tumor Segmentation with Missing Modalities  
This repository is the work of "Modality-adaptive Feature Interaction for Brain Tumor Segmentation with Missing Modalities " based on pytorch implementation.

## Requirements  
All experiments use the PyTorch library. We recommend installing the following package versions:

   python=3.7

   pytorch=1.7.0

   torchvision=0.8.1

## Dateset 
We use multimodal brain tumor dataset (BraTS 2018) in our experiments. We provide txt for data spliting, which is same as [U-HVED](https://github.com/ReubenDo/U-HVED).
### Data preprocess
Convert the .nii files as .pkl files. Normalization with zero-mean and unit variance.  
```Python
python preprocess.py
```
## Training
Our model can be trained by running following command:
```Python
python train.py
```
## Inference
Our model can be tested with a saved model using following command:  
```Python
python test.py
```
The inference code will test all 15 cases with missing modalities together.

## Acknowledge
The implementation is based on the repo: [DMFNet](https://github.com/China-LiuXiaopeng/BraTS-DMFNet)
