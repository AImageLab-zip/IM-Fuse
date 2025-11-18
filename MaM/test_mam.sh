#! /bin/bash
mask_values=(0 1)
mask3=(0 1)
mask4=(0 1)

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