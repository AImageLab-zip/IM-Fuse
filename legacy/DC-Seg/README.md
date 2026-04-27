# MICCAI 2025 Oral: DC-Seg
Official implementation of [DC-Seg: Disentangled Contrastive Learning for Brain Tumor Segmentation with Missing Modalities](https://arxiv.org/abs/2505.11921)
## Environment Set Up
```bash
conda create -n dcseg python=3.10
conda activate dcseg
pip install -r requirements.txt
```
## Exps for Brats
```bash
sh job_brats.sh
```

## Exps for WMH
```bash
sh job_wmh.sh
```

## Data Preparation

Please download the data from the following link:
📦 [Google Drive](https://drive.google.com/drive/folders/1AwLwGgEBQwesIDTlWpubbwqxxd8brt5A)

> Acknowledgment:
The dataset is provided by the RFNet authors. We sincerely thank them for sharing their data resources.

## Model Weights and Training Details

The pre-trained DC-Seg weights, along with a detailed description of one complete training process, have been released and are available on Hugging Face:
👉 https://huggingface.co/cucl2/DC-Seg
