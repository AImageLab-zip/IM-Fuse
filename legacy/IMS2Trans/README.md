# IMS2Trans: Scalable Swin Transformer Network

This repo holds code for **Scalable Swin Transformer Network for Brain Tumor Segmentation from Incomplete MRI Modalities**.



<!-- ## Abstract

Deep learning methods have shown great potential in processing multi-modal Magnetic Resonance Imaging (MRI) data, enabling improved accuracy in brain tumor segmentation. However, the performance of these methods can suffer when dealing with incomplete modalities, which is a common issue in clinical practice. Existing solutions, such as missing modality synthesis, knowledge distillation, and architecture-based methods, suffer from drawbacks such as long training times, high model complexity, and poor scalability. This paper proposes IMS2Trans, a novel lightweight scalable Swin Transformer network by utilizing a single encoder to extract latent feature maps from all available modalities. This unified feature extraction process enables efficient information sharing and fusion among the modalities, resulting in efficiency without compromising segmentation performance even in the presence of missing modalities. Evaluated on a popular benchmark of a brain tumor segmentation (BraTS) dataset with incomplete modalities, our model achieved higher average Dice similarity coefficient (DSC) scores for the whole tumor, tumor core, and enhancing tumor regions (86.57, 75.67, and 58.28, respectively), in comparison with a state-of-the-art model, i.e. mmFormer (86.45, 75.51, and 57.79, respectively). Moreover, our model exhibits significantly reduced complexity with only 4.47M parameters, 121.89G FLOPs, and a model size of 77.13MB, whereas mmFormer comprises 34.96M parameters, 265.79G FLOPs, and a model size of 559.74MB. These indicate our model, being light-weighted with significantly reduced parameters, is still able to achieve better performance than a state-of-the-art model.By leveraging a single encoder for processing the available modalities, IMS2Trans offers notable scalability advantages over methods that rely on multiple encoders. This streamlined approach eliminates the need for maintaining separate encoders for each modality, resulting in a lightweight and scalable network architecture.  -->


## Usage. 

* Environment Preparation
  * Download the python 3.6+ and cuda 9.0+ and pytorch 1.2+.
  * Please use the command `pip install -r requirements.txt` for the dependencies.
  * Set the environment path in `job.sh`.
* Data Preparation
- Download the data from [MICCAI 2018 BraTS Challenge](https://www.med.upenn.edu/sbia/brats2018/data.html).
  - Set the data path in `preprocess.py` and then run `python preprocess.py`.
- Set the data path in `job.sh`
* Train

  - Train the model by `sh job.sh`. 

* Test
  * The trained model should be located in `IMS2Trans/output`. 
  * Uncomment the evaluation command in  `job.sh` and then inference on the test data by `sh job.sh`.
  * The pre-trained model `model_last.pth` located in `IMS2Trans/model` is available.\
    Note that because this model_last.pth file is larger than 50MB, downloading as a ZIP does not include the file, and you'll need to use Git over the command line to get the full objects, or click [model_last.pth](https://github.com/hudscomdz/IMS2Trans/raw/main/model/model_last.pth) to download.

<!-- ## Citation

If you find this code and paper useful for your research, please kindly cite our paper.

```
@article{zhang2023ims2trans,
  title={Scalable Swin Transformer Network for Brain Tumor Segmentation from Incomplete MRI Modalities},
  author={Dongsong Zhang, Changjian Wang, Tianhua Chen, Weidao Chen, and Yiqing Shen},
  journal={Submitted to Artificial Intelligence in Medicine},
  year={2023}
}
``` -->



