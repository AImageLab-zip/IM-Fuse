# A2FSeg: Adaptive Multi-Modal Fusion Network for Medical Image Segmentation

## Paper

This is the implementation for the paper:

[A2FSeg: Adaptive Multi-Modal Fusion Network for Medical Image Segmentation]()

Early Accepted by MICCAI 2023

![image]()

## Usage

* Data Preparation

  - Download the data from [MICCAI 2020 BraTS Challenge](https://www.med.upenn.edu/cbica/brats2020/data.html).

  - Convert the files' name by

  `python dataset_conversion/Task032_BraTS_2020.py`

  - Preprocess the data by

  `python experiment_planning/nnUNet_plan_and_preprocess.py -t 32 --verify_dataset_integrity`

  - Change batchsize by

  `python experiment_planning/change_batch_size.py`
  
* Train

  Train the model by

  `python run/run_training_MAML3_channel.py`

 `A2FSeg` is integrated with the out-of-box [nnUNet](https://github.com/MIC-DKFZ/nnUNet). Please refer to it for more usage.

## Citation

If you find this code and paper useful for your research, please kindly cite our paper.

```

```

## Acknowledgement

`A2FSeg` is integrated with the out-of-box [nnUNet](https://github.com/MIC-DKFZ/nnUNet).
