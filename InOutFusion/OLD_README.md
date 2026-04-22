# InOutFusion: Hierarchical In-Out Fusion for Incomplete Multimodal Brain Tumor Segmentation
  
Our implementation is on an NVIDIA RTX 4090 (24G) with PyTorch 1.9.1.

## Datasets
We use the BraTS2020 dataset, an open-source dataset. 
Please download and unzip the 'MICCAI_BraTS2020_TrainingData' into `./dataset`.
We also use the BraTS2018 dataset, an open-source dataset. 
Download the data from MICCAI 2018 BraTS Challenge.
You can download the BraTS2020 link
https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation.
you can download the BraTS2018 link
https://www.med.upenn.edu/sbia/brats2018/data.html.
Please download and unzip the 'MICCAI_BraTS2020_TrainingData' into `./dataset`.

Set the data path in preprocess.py and then run python.

If you use MICCAI 2018 BraTSdataset,  please run rename18.py before preprocess.py.

Then, please `cd ./process` and run the following commands to prepare the data:
```
python split.py
```

## Training Examples
```
python train.py --phase train --model_name  RsInOut_U_Hemis3D
```
Saved models can be found at `./checkpoint`. 
model_name includes : 'RsInOut_U_Hemis3D' 'TF_U_Hemis3D', 'U_Hemis3D', 'RMBTS', 'TF_RMBTS', 'LMCR', 'TF_LMCR' .


## Test Examples (Please train the model before test.)
```
python train.py --phase test --model_name RsInOut_U_Hemis3D
```
Brain tumor segmentation results for test data can be found at `./checkpoint`. 

## Evaluation
```
python evaluation.py --model_name RsInOut_U_Hemis3D

```


train.py is used for the BraTS2020 dataset; train2018.py is used for the BraTS2018 dataset.
evaluation.py is used for the BraTS2020 dataset; evaluation2018.py is used for the BraTS2018 dataset.
