#!/usr/bin/env python3
# encoding: utf-8
import os
import random
import torch
import warnings
import numpy as np
from losses import Dice
from binary import hd95
import logging

criteria = Dice() 

def init_env(gpu_id='0', seed=42):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings('ignore')

def get_mask(seg_volume):
    seg_volume = seg_volume.detach().cpu().numpy()
    seg_volume = np.squeeze(seg_volume)

    wt_pred = seg_volume[0]
    tc_pred = seg_volume[1]
    et_pred = seg_volume[2]

    wt_mask = np.zeros_like(wt_pred)
    tc_mask = np.zeros_like(tc_pred)
    et_mask = np.zeros_like(et_pred)

    wt_mask[wt_pred > 0.5] = 1
    tc_mask[tc_pred > 0.5] = 1
    et_mask[et_pred > 0.5] = 1

    wt_mask = wt_mask.astype("uint8")
    tc_mask = tc_mask.astype("uint8")
    et_mask = et_mask.astype("uint8")

    masks = [wt_mask, tc_mask, et_mask]
    
    return masks

def get_mask_divide(seg_volume):
    seg_volume = seg_volume.detach().cpu().numpy()
    seg_volume = np.squeeze(seg_volume)

    # [net_volume, snfh_volume, et_volume, bg_volume]
    ncr_pred = seg_volume[0]
    ed_pred = seg_volume[1]
    et_pred = seg_volume[2]

    wt_mask = np.zeros_like(ncr_pred)
    tc_mask = np.zeros_like(ed_pred)
    et_mask = np.zeros_like(et_pred)

    wt_mask = ((ncr_pred > 0.5) | (ed_pred > 0.5) | (et_pred > 0.5)).astype("uint8")
    tc_mask = ((ncr_pred > 0.5) | (et_pred > 0.5)).astype("uint8")
    et_mask[et_pred > 0.5] = 1

    wt_mask = wt_mask.astype("uint8")
    tc_mask = tc_mask.astype("uint8")
    et_mask = et_mask.astype("uint8")

    masks = [wt_mask, tc_mask, et_mask]

    return masks

def eval_metrics(gt, pred):
    score_wt = criteria(np.where(gt[0]==1, 1, 0), np.where(pred[0]==1, 1, 0))
    score_ct = criteria(np.where(gt[1]==1, 1, 0), np.where(pred[1]==1, 1, 0))
    score_et = criteria(np.where(gt[2]==1, 1, 0), np.where(pred[2]==1, 1, 0))
    
    return score_wt, score_et, score_ct

# def measure_dice_score(batch_pred, batch_y):
#     pred = get_mask(batch_pred)
#     gt = get_mask(batch_y)
#     score_wt, score_et, score_ct = eval_metrics(gt, pred)
#     score = (score_wt + score_et + score_ct) / 3.0
    
#     return score_wt, score_et, score_ct

def measure_dice_score(batch_pred, batch_y, divide=False):
    if divide:
        pred = get_mask_divide(batch_pred)
        gt = get_mask_divide(batch_y)
    else:
        pred = get_mask(batch_pred)
        gt = get_mask(batch_y)

    score_wt, score_et, score_ct = eval_metrics(gt, pred)
    # score = (score_wt + score_et + score_ct) / 3.0
    
    return score_wt, score_et, score_ct

def measure_hd95(batch_pred, batch_y, divide=False):

    #对于 whole tumor
    # mask_gt = ((gt == 0)|(gt==1)|(gt==2)).astype(int)
    # mask_pred = ((pred == 0)|(pred==1)|(pred==2)).astype(int)
    if divide:
        batch_pred = get_mask_divide(batch_pred)
        batch_y = get_mask_divide(batch_y)
    else:
        batch_pred = get_mask(batch_pred)
        batch_y = get_mask(batch_y)

    hd95_whole = hd95(batch_pred[0], batch_y[0])
    #对于tumor core
    hd95_core = hd95(batch_pred[1], batch_y[1])
    #对于enhancing tumor
    hd95_enh = hd95(batch_pred[2], batch_y[2])

    return hd95_whole,hd95_enh,hd95_core


# def compute_BraTS_HD95(ref, pred):
#     """
#     计算 HD95 指标，内部进行二值化处理。
#     :param ref: 真实标签（NumPy 数组）
#     :param pred: 预测值（NumPy 数组）
#     :return: HD95 距离
#     """
#     # 二值化处理
#     ref = (ref > 0.5) # 真实标签二值化，阈值 0.5
#     pred = (pred > 0.5)# 预测值二值化，阈值 0.5
#
#     ref=ref.cpu().numpy()
#     pred=pred.cpu().numpy()
#
#     ref = np.squeeze(ref)  # 去掉不必要的维度，确保是 (D, H, W)
#     pred = np.squeeze(pred)
#
#     num_ref = np.sum(ref)
#     num_pred = np.sum(pred)
#
#     # 如果真实或预测为空，按约定返回值
#     if num_ref == 0 and num_pred == 0:
#         return 0.0
#     elif num_ref == 0 or num_pred == 0:
#         return 373.13
#
#     # 计算 HD95
#     return hd95(pred, ref, (1, 1, 1))  # 假设 spacing=(1, 1, 1)

def evaluate_sample(batch_pred_full, batch_pred_missing, batch_y):

    pred_nii_full = get_mask(batch_pred_full)
    pred_nii_miss = get_mask(batch_pred_missing)
    gt_nii = get_mask(batch_y)
    
    metric_full  = eval_metrics(gt_nii, pred_nii_full)
    metric_miss  = eval_metrics(gt_nii, pred_nii_miss)

    hd95_full = measure_hd95(gt_nii, pred_nii_full)
    hd95_miss = measure_hd95(gt_nii, pred_nii_miss)

    return metric_full, metric_miss , hd95_full, hd95_miss

def load_old_model(model_full, model_missing, d_style, optimizer, scheduler, saved_model_path):
    print(f"Constructing model from saved file: {saved_model_path}")
    checkpoint = torch.load(saved_model_path)
    model_full.load_state_dict(checkpoint["model_full"])
    model_missing.load_state_dict(checkpoint["model_missing"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    #optimizer_d_style.load_state_dict(checkpoint["optimizer_d_style"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    d_style.load_state_dict(checkpoint["d_style"])
    epoch = checkpoint["epochs"]
    
    if checkpoint["dice"]:
        dice = checkpoint["dice"]
    else:
        dice = 0.0

    return model_full, model_missing, d_style, optimizer, scheduler, epoch, dice
        
def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def get_logging(log_dir):
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 设置全局日志级别

    # 创建文件处理器，输出到文件
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.INFO)  # 设置文件日志的级别（例如，INFO）

    # 创建控制台处理器，输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # 设置控制台日志的级别（例如，DEBUG）

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger