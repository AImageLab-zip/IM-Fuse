# Missing as Masking: Arbitrary Cross-modal Feature Reconstruction for Incomplete Multimodal Brain Tumor Segmentation
[[Paper]](https://papers.miccai.org/miccai-2024/paper/0067_paper.pdf)  MICCAI 2024
![Missing as Masking overview](/MaM/MaM.png)
## How to run
Clone this repository, create a python env for the project and activate it. Then install all the dependencies with pip
```
cd MaM
python -m venv mam_venv
source mam_venv/bin/activate
pip install -r requirements.txt
```

## Preprocess data
```
nnUNetv2_plan_and_preprocess -d 138 --verify_dataset_integrity
```

## Training
```
python /work/grana_neuro/missing_modalities/nnUNet/nnunetv2/run/run_training.py -tr nnUNetTrainerMissingReconBaseline 138 3d_fullres_multiencoder_recon 0 --c

```

## Test
```
mask_values=(0 1)
mask3=(1)
mask4=(1)

for m1 in ${mask_values[@]}; do
    for m2 in ${mask_values[@]}; do
        for m3 in ${mask3[@]}; do
            for m4 in ${mask4[@]}; do
                if [ "$m1$m2$m3$m4" == "0000" ]; then
                    continue
                fi
                mask="$m1,$m2,$m3,$m4"
                echo "Running nnUNetv2_predict with mask $mask"
                nnUNetv2_predict -i /work/grana_neuro/nnUNet_raw/Dataset138_BraTS2023/imagesTs -o /work/grana_neuro/nnUNet_raw/Dataset138_BraTS2023/prediction_"$m1$m2$m3$m4" -d 138 -tr nnUNetTrainerMissingReconBaseline -c 3d_fullres_multiencoder_recon -f 0 -chk checkpoint_best.pth -mask $mask
            done
        done
    done
done

```