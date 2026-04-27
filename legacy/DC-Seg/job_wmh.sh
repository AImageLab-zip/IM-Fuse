export CUDA_VISIBLE_DEVICES=1
python train.py \
    --batch_size 2 \
    --iter_per_epoch -1 \
    --num_epochs 500 \
    --savepath /mnt/data2/xxx/output/hetero/Mymodel_RFM/wmh_split_person2 \
    --use_recon_loss \
    --use_reg_loss \
    --use_ana_contrastive \
    --use_mod_contrastive \
    --dataname wmh \
    --datapath /data1/xxx/datasets/Zhe_2_MRI/processed