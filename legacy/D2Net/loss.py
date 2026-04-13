import numpy as np
import argparse
import os
import time
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Parser 
from utils.criterions import *
from utils import lovasz_loss as lovasz


class Losses(torch.nn.Module):
    def __init__(self):
        super(Losses, self).__init__()

    def forward(self, pred, target, datasets='BraTSDataset', use_dice=False, no_sigmoid_dice=False, ce=False, focal=False, bce=False, iou=False, ssim=False, use_lovasz=False, use_TverskyLoss=False):
        loss = 0.0
        if use_dice:
            dice_loss = sigmoid_dice_loss(pred, target)
            loss = loss + dice_loss
            #print ('Dice loss: %f |'%dice_loss.item(), end = " ")
        if no_sigmoid_dice:
            no_sig_dice_loss = dice(pred.clone(), target.clone())
            loss = loss + no_sig_dice_loss 
            #print ('NoSigmoid_Dice loss: %f |'%no_sig_dice_loss.item(), end = " ")
        if ce:
            ce_loss = CE_loss(pred, target)
            loss = loss + ce_loss
            #print ('CE loss: %f |'%ce_loss.item(), end = " ")
        if focal:
            focal_loss = FocalLoss(pred, target, alpha=0.25, gamma=2.0)
            loss = loss + focal_loss
            #print ('Focal loss: %f |'%focal_loss.item())
        if bce:
            bce_loss = bce_loss(pred.clone(), target.clone())
            loss = loss + bce_loss
            #print ('BCE loss: %f |'%bce_loss, end = " ")
        if iou:
            iou_loss = IOU_loss(pred.clone(), target.clone())
            loss = loss + iou_loss
            #print ('IOU loss: %f |'%iou_loss, end = " ")
        if ssim:
            _ssim_loss = SSIM()
            ssim_loss = _ssim_loss(pred.clone(), target.clone())
            loss = loss + ssim_loss
            #print ('SSIM loss: %f |'%ssim_loss, end = " ")
        if use_lovasz:
            lovasz_loss = lovasz.lovasz_softmax(pred.clone(), target.clone())
            loss = loss + lovasz_loss
            #print ('Lovasz loss: %f |'%lovasz_loss, end = " ")
        if use_TverskyLoss:
            Tversky_loss = TverskyLoss(pred.clone(), target.clone())
            loss = loss + Tversky_loss
            #print ('Tversky loss: %f |'%Tversky_loss, end = " ")
        return loss

class Loss_Region(torch.nn.Module):
    def __init__(self):
        super(Loss_Region, self).__init__()

    def forward(self, pred, target, use_dice=False, ce=False, focal=False, bce=True, iou=False, ssim=False, use_lovasz=False, use_TverskyLoss=False):
        loss = 0.0
        output_bg = pred[:,0,...]
        output_et = pred[:,3,...]
        output_tc = pred[:,1,...] + pred[:,3,...]
        output_wt = pred[:,1,...] + pred[:,2,...] + pred[:,3,...]

        b,h,w,d = target.shape
        target_region = target.clone().expand(4,b,h,w,d).clone()
        target_region = target_region.permute(1,0,2,3,4)
        
        target_region[:,0,...] = (target==0)
        target_region[:,1,...] = (target==4)
        target_region[:,2,...] = (target==1) + (target==4)
        target_region[:,3,...] = (target==1) + (target==2) + (target==4)

        if bce:
            _output_bg = F.sigmoid(output_bg)
            _output_et = F.sigmoid(output_et)
            _output_tc = F.sigmoid(output_tc) 
            _output_wt = F.sigmoid(output_wt)

            bce_loss_bg = bce_loss(_output_bg.clone(), target_region[:,0,...].cuda())
            bce_loss_et = bce_loss(_output_et.clone(), target_region[:,1,...].cuda())
            bce_loss_tc = bce_loss(_output_tc.clone(), target_region[:,2,...].cuda())
            bce_loss_wt = bce_loss(_output_wt.clone(), target_region[:,3,...].cuda())

            _bce_loss = (bce_loss_et + bce_loss_tc + bce_loss_wt )/3
            loss = loss + _bce_loss
            print ('Region_BCE loss: %f |'%_bce_loss.item(), end = " ")
        
        if use_dice:
            dice_loss_bg = dice(F.sigmoid(output_bg), target_region[:,0,...].cuda()==1)
            dice_loss_et = dice(F.sigmoid(output_et), target_region[:,1,...].cuda()==1)
            dice_loss_tc = dice(F.sigmoid(output_tc), target_region[:,2,...].cuda()==1)
            dice_loss_wt = dice(F.sigmoid(output_wt), target_region[:,3,...].cuda()==1)
            
            _dice_loss = (dice_loss_et + dice_loss_tc + dice_loss_wt)/3
            loss = loss + _dice_loss 
            print ('Region_Dice loss: %f |'%_dice_loss.item(), end = " ")
        
        del target_region
        torch.cuda.empty_cache()

        return loss
