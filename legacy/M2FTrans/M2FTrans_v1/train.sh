#!/bin/bash
pythonname='/home/sjj/anaconda3/envs/m2ftrans'

# dataname='BRATS2020'
dataname='/home/sjj/M2FTrans/BraTS/BRATS2018'
pypath=$pythonname
cudapath='/home/sjj/anaconda3'
datapath=${dataname}_Training_none_npy
savepath='output/m2ftrans_batchsize4_iter150_epoch1000_lr2e-4_rfse0'
resume='/home/sjj/M2FTrans/M2FTrans_v1/output/m2ftrans_batchsize4_iter150_epoch1000_lr2e-4_rfse0/model_last.pth'
model='fusiontrans'

export PATH=$cudapath/bin:$PATH
export LD_LIBRARY_PATH=$cudapath/lib:$LD_LIBRARY_PATH
PYTHON=$pypath/bin/python3.8
export PATH=$pypath/include:$pypath/bin:$PATH
export LD_LIBRARY_PATH=$pypath/lib:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0,1
#train:
# without pretrain
$PYTHON train.py --model $model --batch_size=4 --num_epochs 1000 --iter_per_epoch 150 --lr 2e-4 --region_fusion_start_epoch 0 --dataname $dataname --datapath $datapath --savepath $savepath
# with pretrain
# $PYTHON train.py --model $model --batch_size=4 --num_epochs 1000 --iter_per_epoch 150 --lr 2e-4 --region_fusion_start_epoch 0 --savepath $savepath --dataname $dataname --datapath $datapath --resume $resume


