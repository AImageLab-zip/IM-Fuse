#!/bin/bash

dataname='BRATS2018'
datapath=BRATS18/${dataname}_Training_none_npy
savepath=output
 
export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py --batch_size=1 --datapath $datapath --savepath $savepath --num_epochs 1000 --dataname $dataname

##eval:
#resume=output/model_last.pth
#python train.py --batch_size=1 --datapath $datapath --savepath $savepath --num_epochs 0 --dataname $dataname --resume $resume