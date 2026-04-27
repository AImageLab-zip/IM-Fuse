export CUDA_VISIBLE_DEVICES=0
python train.py \
    --batch_size 2 \
    --iter_per_epoch -1 \
    --num_epochs 500 \
    --fusion_type RFM \
    --use_reg_loss \
    --use_recon_loss \
    --use_ana_contrastive \
    --use_mod_contrastive \
    --crop_size 112 \
    --savepath /mnt/data2/xxx/output/hetero/Mymodel_RFM/Brats2020_contrastive_112 \
    --datapath /data1/xxx/datasets/RFNet/BRATS2020_Training_none_npy