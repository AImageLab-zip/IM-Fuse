# The code is extensively uses the ACN implementation, please see:
## https://github.com/Wangyixinxin/ACN##
#!/usr/bin/env python3
# encoding: utf-8
import yaml
from data import make_data_loaders_divide_2023, make_data_loaders_divide
from models import build_MSTKDNet
from losses import Dice
import os
import torch
from utils.utils import *
from tqdm import tqdm
from utils.random_seed import setup_seed
from models.discriminator import get_style_discriminator
from solver import make_optimizer_double

a=["flair",'t1','t1ce','t2']

masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]

mask_name = ['t2', 't1', 't1c', 'flair',
            't1t2', 't1cet1', 'flairt1ce', 't1cet2', 'flairt2', 'flairt1',
            'flairt1cet1', 'flairt1cet2', 'flairt1t2', 't1cet1cet2',
            'flairt1cet1t2']

    
def test_val(model_missing, loaders, i, mask_name):
    for epoch in range(0, 1):
        dice_wt = 0.0
        dice_et = 0.0
        dice_tc = 0.0
        dice = 0.0

        for phase in ['test']:
            loader = loaders[phase]
            total = len(loader)
            for batch_id, (batch_x, batch_y, mask) in tqdm(enumerate(loader), total=total, desc="Training Batches"):
                # iter_num = iter_num + 1
                batch_x, batch_y = batch_x.cuda(non_blocking=True), batch_y.cuda(non_blocking=True)

                with torch.set_grad_enabled(False):
                    mask = masks_test[i]
                    mask_tensor = torch.tensor(mask, dtype=torch.float32).view(1, 4, 1, 1, 1).cuda(non_blocking=True)  # (B, 4, 1, 1, 1)
                    batch_xn = batch_x * mask_tensor                                # (B, 4, 160, 192, 128)

                    seg_m, style_m, content_m, unetr_fs_m, att_w_m, Gs_m, logit_m = model_missing(batch_xn[:, 0:])

                    wt, et, tc = measure_dice_score(seg_m, batch_y, divide=True)
                    dice += (wt + et + tc) / 3.0
                    dice_wt += wt
                    dice_et += et
                    dice_tc += tc

            dice = (dice / (batch_id + 1))
            print(
                f'Epoch {epoch + 1} validation dice score for  modality {mask_name}>> {dice_wt / (batch_id + 1)} ,core{dice_tc / (batch_id + 1)},enhance{dice_et / (batch_id + 1)}')

if __name__ == '__main__':
    ## Main section    
    ########## print config
    config = yaml.load(open(os.path.join('./configs', ('brats.yml'))), Loader=yaml.FullLoader)
    for k, v in config.items():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True)

    setup_seed(config["seed"])

    ########## Setting data
    if config['dataset'] == 'brats23':
        train_file = 'datalist/train.txt'
        test_file = 'datalist/test15splits2.csv'
        val_file = 'datalist/val15splits.csv'
        loaders = make_data_loaders_divide_2023(config, train_list_path=train_file, val_list_path=val_file, test_list_path=test_file)
    elif config['dataset'] == 'brats18':
        train_file = 'datalist/Brats18_train.csv'
        test_file = 'datalist/Brats18_test.csv'
        val_file = 'datalist/Brats18_val.csv'
        loaders = make_data_loaders_divide(config, train_list_path=train_file, val_list_path=val_file, test_list_path=test_file)
    else:
        raise NotImplementedError
    
    model_full, model_missing = build_MSTKDNet(inp_dim1 = 4, inp_dim2 = 4)
    model_full    = model_full.cuda()
    model_missing = model_missing.cuda()
    d_style       = get_style_discriminator(num_classes = 128).cuda()

    #slurm_job_id = os.getenv("SLURM_JOB_ID")
    #log_dir = os.path.join(config['path_to_log'], 'brats23'+ str(slurm_job_id))
    #if not os.path.exists(log_dir):
    #    os.makedirs(log_dir)  
        
    criteria = Dice() 
    #saved_model_path = log_dir+'/model_best.pth'
    #print(f'Reading the model from path: {saved_model_path} ')

    optimizer, scheduler = make_optimizer_double(config, model_full, model_missing)
    
    latest_model_path = os.path.join(config['path_to_latest_model'], 'model_best.pth')
    print(f'Reading the model from path: {latest_model_path} ')
    
    model_full, model_missing, d_style, optimizer, scheduler, epoch, best_dice = load_old_model(model_full, model_missing, d_style, optimizer, scheduler, latest_model_path)

    for i in range(0,15):
        test_val(model_missing, loaders, i, mask_name[i])

    print('Test process is finished')
