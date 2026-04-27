source /work/grana_neuro/missing_modalities/mmformer_venv/bin/activate

python -m debugpy --listen 0.0.0.0:5678 --wait-for-client \
    /work/grana_neuro/missing_modalities/DC-Seg/train.py \
    --batch_size 1 \
    --dataname BRATS2023 \
    --iter_per_epoch -1 \
    --num_epochs 500 \
    --fusion_type RFM \
    --use_reg_loss \
    --use_recon_loss \
    --use_ana_contrastive \
    --use_mod_contrastive \
    --crop_size 112 \
    --savepath /work/grana_neuro/missing_modalities/DC-Seg/output_dbg \
    --datapath /work/grana_neuro/missing_modalities/BRATS2023_Training_npy \
    --debug