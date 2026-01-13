import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms as T
from Model import no_share_unet
from transforms import *
from Ddataset import BraTSDataset
from glob import glob
from losses import *
import torchvision
#from evaluation import *
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from metrics import dice_coef
from tqdm import tqdm
import argparse
from pathlib import Path
import wandb
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')

    # datapath and dataset
    #parser.add_argument('--train_list', type=str, default='train1.txt')
    #parser.add_argument('--val_list', type=str, default='val1.txt')
    
    parser.add_argument('--datapath', type=Path, required=True)
    parser.add_argument('--wandb-project-name',type=str,default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--checkpoint-path',type=Path,required=True)
    parser.add_argument('--num-epochs', type=int, default=1200)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--train_transforms', type=str, default='Compose([ RandCrop3D((128,128,128)), RandomRotion(10),RandomFlip(0), NumpyType((np.float32, np.int64)), ])')
    parser.add_argument('--val_transforms', type=str, default='Compose([Pad((0, 0, 0, 5, 0)),NumpyType((np.float32, np.int64)),])')
    opt = parser.parse_args()
    return opt


def adjust_lr(init_lr,optimizer, epoch,total_epo):
    cur_lr = init_lr * (1-epoch/total_epo)**0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr 

def train_model(model, criterion, optimizer, dataload, val_loader, scaler, sche=None,num_epochs=1200,
                deepSupvision=False,wandb_project_name=None,checkpoint_path=None,lr=5e-5,
                start_epoch = 0):
    if wandb_project_name is not None:
        wandb.init(project=wandb_project_name,name='training')

    #use_amp = torch.cuda.get_device_capability(0)[0]>=7
    use_amp = False
    if use_amp:
        print('Using automatic mixed precision!')

    best_acc = 0.0
    dt_size = len(dataload.dataset)
    num_steps = (dt_size - 1) // dataload.batch_size + 1
    for epoch in range(start_epoch,num_epochs):
        model.train()
        adjust_lr(init_lr=lr,optimizer=optimizer,epoch=epoch,total_epo=num_epochs)
        
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x1,x2,x3,x4,y,mask_code in tqdm(dataload,total=len(dataload),desc=f'Training epoch{epoch}'):
            step += 1
            input1,input2,input3,input4 = x1.cuda(),x2.cuda(),x3.cuda(),x4.cuda()
            labels = y.cuda()
            optimizer.zero_grad()


            mask_code = mask_code.cuda()
            if use_amp:
                with autocast('cuda'):
                    allOut =  model(input1,input2,input3,input4,mask_code)
                    loss= 0
                    step_loss = []
                    for out in allOut:
                        cur_loss = criterion(out, labels)
                        loss += cur_loss
                        step_loss.append(cur_loss.item())
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            else:
                allOut =  model(input1,input2,input3,input4,mask_code)
                loss= 0
                step_loss = []
                for out in allOut:
                    cur_loss = criterion(out, labels)
                    loss += cur_loss
                    step_loss.append(cur_loss.item())
                loss.backward()
                optimizer.step()
            epoch_loss += step_loss[-1]

        avg_epoch_loss = epoch_loss / num_steps
        lr = optimizer.param_groups[0]["lr"]

        if wandb_project_name is not None:
            wandb.log(
                {
                    "train/epoch": epoch + 1,
                    "train/loss": avg_epoch_loss,
                    "train/lr": lr,
                },
                step=epoch + 1
            )
        else:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"LR: {lr:.6f} "
                f"Loss: {avg_epoch_loss:.6f}"
            )

        ###evaluate model every 5 epoch
        if epoch % 5 ==0:
            H, W, T = 240, 240, 155
            WT_dice, TC_dice, ET_dice = [], [], []

            model.eval()
            with torch.no_grad():
                with autocast('cuda',enabled=use_amp):
                    for i, (x1, x2, x3, x4, target, mask) in tqdm(enumerate(val_loader), total=len(val_loader),desc=f'Eval epoch {epoch}'):
                        mask = mask.cuda()

                        input_x11 = x1[:, :, :120, :120, :]
                        input_x12 = x1[:, :, :120, 120:, :]
                        input_x13 = x1[:, :, 120:, :120, :]
                        input_x14 = x1[:, :, 120:, 120:, :]
                        input_x11, input_x12, input_x13, input_x14 = input_x11.cuda(), input_x12.cuda(), input_x13.cuda(), input_x14.cuda()

                        input_x21 = x2[:, :, :120, :120, :]
                        input_x22 = x2[:, :, :120, 120:, :]
                        input_x23 = x2[:, :, 120:, :120, :]
                        input_x24 = x2[:, :, 120:, 120:, :]
                        input_x21, input_x22, input_x23, input_x24 = input_x21.cuda(), input_x22.cuda(), input_x23.cuda(), input_x24.cuda()

                        input_x31 = x3[:, :, :120, :120, :]
                        input_x32 = x3[:, :, :120, 120:, :]
                        input_x33 = x3[:, :, 120:, :120, :]
                        input_x34 = x3[:, :, 120:, 120:, :]
                        input_x31, input_x32, input_x33, input_x34 = input_x31.cuda(), input_x32.cuda(), input_x33.cuda(), input_x34.cuda()

                        input_x41 = x4[:, :, :120, :120, :]
                        input_x42 = x4[:, :, :120, 120:, :]
                        input_x43 = x4[:, :, 120:, :120, :]
                        input_x44 = x4[:, :, 120:, 120:, :]
                        input_x41, input_x42, input_x43, input_x44 = input_x41.cuda(), input_x42.cuda(), input_x43.cuda(), input_x44.cuda()

                        output1 = model(input_x11, input_x21, input_x31, input_x41, mask)[-5:]
                        output2 = model(input_x12, input_x22, input_x32, input_x42, mask)[-5:]
                        output3 = model(input_x13, input_x23, input_x33, input_x43, mask)[-5:]
                        output4 = model(input_x14, input_x24, input_x34, input_x44, mask)[-5:]
                        output1 = (output1[0] + output1[1] + output1[2] + output1[3] + output1[4]) / 5
                        output2 = (output2[0] + output2[1] + output2[2] + output2[3] + output2[4]) / 5
                        output3 = (output3[0] + output3[1] + output3[2] + output3[3] + output3[4]) / 5
                        output4 = (output4[0] + output4[1] + output4[2] + output4[3] + output4[4]) / 5
                        outputs_half1 = torch.cat((output1, output2), dim=3)
                        outputs_half2 = torch.cat((output3, output4), dim=3)
                        outputs = torch.cat((outputs_half1, outputs_half2), dim=2)
                        outputs = torch.sigmoid(outputs)

                        output = outputs[0, :, :H, :W, :T].cpu().numpy()
                        target = target[0, :, :H, :W, :T].numpy()
                        WT_out = output[0, ...]
                        WT_out[WT_out > 0.5] = 1
                        WT_out[WT_out < 0.5] = 0

                        TC_out = output[1, ...]
                        TC_out[TC_out > 0.5] = 1
                        TC_out[TC_out < 0.5] = 0

                        ET_out = output[2, ...]  # 240,240,155
                        ET_out[ET_out > 0.5] = 1
                        ET_out[ET_out < 0.5] = 0

                        WT_label = target[0, ...]
                        TC_label = target[1, ...]
                        ET_label = target[2, ...]  # 240,240,155
                        wt_dice = dice_coef(WT_out, WT_label)
                        et_dice = dice_coef(ET_out, ET_label)
                        tc_dice = dice_coef(TC_out, TC_label)
                        WT_dice.append(wt_dice)
                        ET_dice.append(et_dice)
                        TC_dice.append(tc_dice)

                mean_wt,mean_tc,mean_et = np.mean(WT_dice),np.mean(TC_dice),np.mean(ET_dice)
                print('WT Dice: %.4f' % mean_wt)
                print('TC Dice: %.4f' % mean_tc)
                print('ET Dice: %.4f' % mean_et)
            acc = (mean_wt+mean_tc+mean_et)/3
            if wandb_project_name is not None:
                wandb.log({"val/mean_wt": mean_wt, 
                           "val/mean_tc": mean_tc,
                           "val/mean_et":mean_et
                       }, step=epoch+1)
            else:
                print(
                    f"Epoch {epoch} | "
                    f"val/mean_wt: {mean_wt:.6f} | "
                    f"val/mean_tc: {mean_tc:.6f} | "
                    f"val/mean_et: {mean_et:.6f}"
                )
            if acc>best_acc:
                print(f'Best score on epoch {epoch}, acc --> {acc}')
                best_acc = acc
        ###save model parameters every 50 epoch when the epoch>800
        if (epoch + 1) % 50 ==0:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler':scaler.state_dict()
            }
            torch.save(state, checkpoint_path / f'chk_{epoch}.pth')
            torch.save(state, checkpoint_path / f'last.pth')
        if sche:
            sche.step()

    return model


# train the model
def train():
    opt = parse_option()
    train_list = Path(__file__).parent / 'datalist' / 'train.txt'
    val_list = Path(__file__).parent / 'datalist' / 'val15splits.csv'
    model = no_share_unet(in_channel=1, out_channel=3, diff=True,deepSupvision=True).cuda()
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(),lr=opt.learning_rate * np.sqrt(opt.batch_size),weight_decay=opt.weight_decay)
    scaler = GradScaler()
    start_epoch = 0
    if opt.resume:
        checkpoint = torch.load(opt.checkpoint_path / f'last.pth',weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] +1
        scaler.load_state_dict(checkpoint['scaler'])
    train_set = BraTSDataset(train_list, root=opt.datapath, mode='train',
                             transforms=opt.train_transforms)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=True, 
        num_workers=opt.num_workers)

    val_set = BraTSDataset(val_list, root=opt.datapath, mode='val',
                            transforms=opt.val_transforms)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=1,
        pin_memory=True, 
        num_workers=opt.num_workers)

    train_model(model, criterion, optimizer, train_loader,val_loader,scaler = scaler,sche=None,num_epochs=opt.num_epochs,
                deepSupvision=True,wandb_project_name=opt.wandb_project_name,checkpoint_path=opt.checkpoint_path,
                lr=opt.learning_rate,start_epoch=start_epoch)
    if opt.wandb_project_name is not None:
        wandb.finish()



if __name__ == '__main__':
    train()



