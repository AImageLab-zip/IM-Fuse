# The code is extensively uses the ACN implementation, please see:
## https://github.com/Wangyixinxin/ACN##
#!/usr/bin/env python3
# encoding: utf-8
import yaml
from data import make_data_loaders
from models import build_model, build_smunetr
from models.discriminator import get_style_discriminator
from solver import make_optimizer_double
from losses import get_losses, bce_loss, Dice, get_current_consistency_weight
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils.utils import *
import nibabel as nib
import numpy as np
from tqdm import tqdm
a=["flair",'t1','t1ce','t2']
masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]

mask_name = ['t2', 't1', 't1c', 'flair',
            't1t2', 't1cet1', 'flairt1ce', 't1cet2', 'flairt2', 'flairt1',
            'flairt1cet1', 'flairt1cet2', 'flairt1t2', 't1cet1cet2',
            'flairt1cet1t2']

## Main section
config = yaml.load(open('./config.yml'), Loader=yaml.FullLoader)

def train_val(model_full, model_missing, d_style, loaders, optimizer, scheduler, losses, log_dir, epoch_init=1, best_dice=0.0, pretrain_full_path=None):
    n_epochs = int(config['epochs'])
    iter_num = 0
    if pretrain_full_path:
        pretrain_full_point = torch.load(pretrain_full_path)
        model_full.load_state_dict(pretrain_full_point["model_full"])
        for param in model_full.parameters():
            param.requires_grad = False
        
        test_val()

    for epoch in range(epoch_init, n_epochs):
        weight_consistency = get_current_consistency_weight(epoch)
        scheduler.step()
        train_loss = 0.0

        dice_wt=0.0
        dice_et=0.0
        dice_tc=0.0
        hd95_wt=0.0
        hd95_et=0.0
        hd95_tc=0.0
        hd95=0.0
        dice=0.0
        
        for phase in ['train', 'eval']:
            if phase == 'eval' and epoch%10!=0 :
                continue
            loader = loaders[phase]
            total = len(loader)
            for batch_id, (batch_x, batch_y) in tqdm(enumerate(loader), total=total, desc="Training Batches"):
                iter_num = iter_num + 1
                batch_x, batch_y = batch_x.cuda(non_blocking=True), batch_y.cuda(non_blocking=True)
                with torch.set_grad_enabled(phase == 'train'):

                    #随机缺失模态
                    rb = random.randint(0, 14)
                    mask = masks_test[rb]
                    mask_tensor = torch.tensor(mask, dtype=torch.float32).view(1, 4, 1, 1, 1)
                    mask_tensor = mask_tensor.cuda(non_blocking=True)
                    batch_xn = batch_x * mask_tensor

                    seg_f, style_f, content_f,  unetr_fs_f, att_w_f = model_full(batch_x[:,0:])
                    seg_m, style_m, content_m, unetr_fs_m, att_w_m = model_missing(batch_xn[:,0:])
                    loss_dict = losses['co_loss'](config, seg_f, content_f, batch_y, seg_m, content_m, style_f, style_m, epoch)
                    unnetr_loss = losses['unetr_loss'](unetr_fs_f, unetr_fs_m)
                    evd_loss = losses['evd_loss'](att_w_f, att_w_m)
                    loss_dict['loss_Co'] += (unnetr_loss * 0.2 + 1e8 * evd_loss)
                    print(f"batch-{batch_id}-CoLoss: {loss_dict['loss_Co']}")
                    print(f"batch-{batch_id}-full_DiceLoss: {loss_dict['loss_dc']} * {weight_full} = {loss_dict['loss_dc'] * weight_full}")
                    print(f"batch-{batch_id}-missing_DiceLoss: {loss_dict['loss_miss_dc']} * {weight_missing} = {loss_dict['loss_miss_dc'] * weight_missing}")
                    print(f"batch-{batch_id}-consisLoss: {loss_dict['consistency_loss']} * {weight_consistency} = {loss_dict['consistency_loss'] * weight_consistency}")
                    print(f"batch-{batch_id}-content Loss: {loss_dict['content_loss']} * {weight_content} = {loss_dict['content_loss'] * weight_content}")
                    print(f"batch-{batch_id}-unnetrLoss: {unnetr_loss} * 0.2 = {unnetr_loss * (0.2)}")
                    print(f"batch-{batch_id}-EVDLoss: {evd_loss} * 10^8 = {evd_loss * (1e8)}")
                    
                    d_style.train()
                    optimizer_d_style = optim.Adam(d_style.parameters(), lr = float(config['lr']), betas=(0.9, 0.99))
                    # labels for style adversarial training
                    source_label = 0
                    target_label = 1

                    optimizer.zero_grad()
                    optimizer_d_style.zero_grad()
                    
                    # only train. Don't accumulate grads in disciminators
                    for param in d_style.parameters():
                        param.requires_grad = False

                    if phase == 'train':
                        (loss_dict['loss_Co']).backward(retain_graph=True)
                        train_loss += loss_dict['loss_Co'].item()
                    
                    ##################### adversarial training ot fool the discriminator ######################
                    df_src_main = style_f
                    df_trg_main = style_m
                    d_df_out_main = d_style(df_trg_main)
                    loss_adv_df_trg_main = bce_loss(d_df_out_main, source_label)
                    loss = 0.0002 * loss_adv_df_trg_main
                    if phase == 'train':
                        loss.backward()                    
                    
                    ####################### Train discriminator networks ######################################
                    # enable training mode on discriminator networks
                    for param in d_style.parameters():
                        param.requires_grad = True
                    df_src_main = df_src_main.detach()
                    d_df_out_main = d_style(df_src_main)
                    loss_d_feature_main = bce_loss(d_df_out_main, source_label)
                    if phase == 'train':
                        loss_d_feature_main.backward()
                    
                    ####################### train with target ##################################################
                    df_trg_main = df_trg_main.detach()
                    d_df_out_main = d_style(df_trg_main)
                    loss_d_feature_main = bce_loss(d_df_out_main, target_label)

                    if phase == 'train':
                        loss_d_feature_main.backward()

                
                num_classes = 4

                if phase == 'train':
                    optimizer.step()
                    optimizer_d_style.step()
                    if (batch_id + 1) % 20 == 0:
                        print(f'Epoch {epoch+1}>> itteration {batch_id+1}>> training loss>> {train_loss/(batch_id+1)}')
 
                else:
                    wt,et,tc=measure_dice_score(seg_m, batch_y)
                    dice+=(wt+et+tc)/3.0
                    dice_wt+=wt
                    dice_et+=et
                    dice_tc+=tc

                    hdwt,hdet,hdtc= measure_hd95(seg_m, batch_y)
                    hd95+=(hdwt+hdet+hdtc)/3.0
                    hd95_wt+=hdwt
                    hd95_et+=hdet
                    hd95_tc+=hdtc

            if phase == 'train':
                print(f'Epoch {epoch+1} overall training loss>> {train_loss/(batch_id+1)}')
                state = {}
                state['model_full'] = model_full.state_dict()
                state['model_missing'] = model_missing.state_dict()
                state['d_style'] = d_style.state_dict()
                state['optimizer'] = optimizer.state_dict()
                state['epochs'] = epoch
                state['dice'] = dice
                file_name = log_dir + '/model_last.pth'
                torch.save(state, file_name)
            else:
                dice = (dice/(batch_id+1))
                hd95 = (hd95/(batch_id+1))
                print(f'Epoch {epoch+1} validation dice score for missing modality whole>> {dice_wt/(batch_id+1)} ,core{dice_tc/(batch_id+1)},enhance{dice_et/(batch_id+1)}')
                print(f'Epoch {epoch+1} validation hd95 score for missing modality whole>> {hd95_wt/(batch_id+1)} ,core{hd95_tc/(batch_id+1)},enhance{hd95_et/(batch_id+1)}')
                if dice > best_dice:
                    print('Get the Best Dice Score !!!')
                    state = {}
                    state['model_full'] = model_full.state_dict()
                    state['model_missing'] = model_missing.state_dict()
                    state['d_style'] = d_style.state_dict()
                    state['optimizer'] = optimizer.state_dict()
                    state['epochs'] = epoch
                    state['dice'] = dice
                    file_name = log_dir+'/model_best.pth'
                    torch.save(state, file_name)
                    best_dice = dice


def test_val(model_full, model_missing, loaders, rb):
    iter_num = 0
    for epoch in range(0,1):
        train_loss = 0.0

        dice_wt = 0.0
        dice_et = 0.0
        dice_tc = 0.0
        hd95_wt = 0.0
        hd95_et = 0.0
        hd95_tc = 0.0
        hd95 = 0.0
        dice = 0.0

        for phase in ['eval']:
            if phase == 'eval' and epoch % 10 != 0:
                continue
            loader = loaders[phase]
            total = len(loader)
            for batch_id, (batch_x, batch_y) in tqdm(enumerate(loader), total=len(loader), desc="test Batches"):
                iter_num = iter_num + 1
                batch_x, batch_y = batch_x.cuda(non_blocking=True), batch_y.cuda(non_blocking=True)
                with torch.set_grad_enabled(phase == 'train'):

                    # 固定缺失模态
                    mask = masks_test[rb]
                    mask_tensor = torch.tensor(mask, dtype=torch.float32).view(1, 4, 1, 1, 1)
                    mask_tensor = mask_tensor.cuda(non_blocking=True)
                    batch_xn = batch_x * mask_tensor
                    seg_m, _, _, _, _ = model_missing(batch_xn[:, 0:])

                wt, et, tc = measure_dice_score(seg_m, batch_y)
                dice += (wt + et + tc) / 3.0
                dice_wt += wt
                dice_et += et
                dice_tc += tc

                hdwt, hdet, hdtc = measure_hd95(seg_m, batch_y)
                hd95 += (hdwt + hdet + hdtc) / 3.0
                hd95_wt += hdwt
                hd95_et += hdet
                hd95_tc += hdtc
            dice = (dice/(batch_id+1))
            hd95 = (hd95/(batch_id+1))
            print(f'score type: dice, modality: {mask_name[rb]}, score: {dice_wt/(batch_id+1)}, core score: {dice_tc/(batch_id+1)}, enhance score: {dice_et/(batch_id+1)}')
            print(f'score type: hd95, modality: {mask_name[rb]}, score: {hd95_wt/(batch_id+1)}, core score: {hd95_tc/(batch_id+1)}, enhance score: {hd95_et/(batch_id+1)}')
            