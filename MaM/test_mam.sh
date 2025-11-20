#! /bin/bash

INPUT_PATH="" #/work/grana_neuro/nnUNet_raw/Dataset138_BraTS2023/imagesTs/
OUTPUT_PATH="" #/work/grana_neuro/nnUNet_raw/Dataset138_BraTS2023/
CHECKPOINT_PATH="" #checkpoint_best.pth

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-path)
      INPUT_PATH="$2"
      shift 2
      ;;
    --output-path)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --checkpoint-path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 --input-path <path> --output-path <path> --checkpoint-path <path>"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Use --help for usage instructions."
      exit 1
      ;;
  esac
done

if [[ -z "$INPUT_PATH" || -z "$OUTPUT_PATH" || -z "$CHECKPOINT_PATH" ]]; then
  echo "Error: Missing required arguments."
  echo "Usage: $0 --input-path <path> --output-path <path> --checkpoint-path <path>"
  exit 1
fi


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
                nnUNetv2_predict -i $INPUT_PATH -o "${OUTPUT_PATH}/prediction_${m1}${m2}${m3}${m4}" -d 138 -tr nnUNetTrainerMissingReconBaseline -c 3d_fullres_multiencoder_recon -f 0 -chk $CHECKPOINT_PATH -mask $mask
            done
        done
    done
done