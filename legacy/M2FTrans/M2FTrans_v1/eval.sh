#!/bin/bash
pythonname='/home/sjj/miniconda3/envs/m2ftrans'

# dataname='BRATS2020'
dataname='/home/sjj/M2FTrans/BraTS/BRATS2018'
pypath=$pythonname
cudapath='/home/sjj/miniconda3'
datapath=${dataname}_Training_none_npy
savepath='output/eval_m2ftrans_batchsize4_iter150_epoch1000_lr2e-4_rfse0'
resume='/home/sjj/M2FTrans/M2FTrans_v1/output/m2ftrans_batchsize4_iter150_epoch1000_lr2e-4_rfse0/model_last.pth'
model='fusiontrans'

export PATH=$cudapath/bin:$PATH
export LD_LIBRARY_PATH=$cudapath/lib:$LD_LIBRARY_PATH
PYTHON=$pypath/bin/python3.8
export PATH=$pypath/include:$pypath/bin:$PATH
export LD_LIBRARY_PATH=$pypath/lib:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0

#eval:

$PYTHON eval.py --batch_size=1 --datapath $datapath --savepath $savepath --dataname $dataname --resume $resume --model $model

