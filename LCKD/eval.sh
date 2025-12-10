#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=Eval_Random_Missing_BraTS18_[80,160,160]_SGD_b2_lr-2_multiTeacher_wt.1

CUDA_VISIBLE_DEVICES=$1 python eval.py \
--input_size=80,160,160 \
--num_classes=3 \
--data_list=BraTS18/BraTS18_test.csv \
--weight_std=True \
--restore_from=snapshots/Random_Missing_BraTS18_[80,160,160]_SGD_b2_lr-2_multiTeacher_wt.1/final.pth \
--mode=1 > logs/${time}_train_${name}.log 2>&1 &
