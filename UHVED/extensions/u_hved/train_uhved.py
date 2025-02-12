#coding=utf-8
import argparse
import os
import time
import logging
import numpy as np
import wandb
import torch
import torch.optim
import sys
from tensorboardX import SummaryWriter
from utils.utils import setup_seed
from u_hved_net_torch import U_HVED
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii, Brats_loadall_val_nii
from data.data_utils import init_fn

from utils.criterions import softmax_dice_loss, cross_loss, compute_KLD, dice_loss, softmax_weighted_loss, softmax_loss
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader, LR_Scheduler_polinomial
from utils.predict import AverageMeter, test_softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default=None, type=str)
parser.add_argument('--dataname', default='BRATS2023', type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--seed', default=999, type=int)
parser.add_argument('--debug', action='store_true', default=False)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((112,112,112)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
MODALITIES_img = ['T1', 'T1c', 'T2', 'Flair']

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))
val_check = [5, 10, 15, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 90, 100, 110, 125, 150, 175, 200, 225, 250, 260, 270, 275, 280, 285, 290, 295, 300] 
print(f"Validation checks: {val_check}")

def main():
    ##########setting seed
    setup_seed(args.seed)
    
    ##########print args
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True)

    ##########init wandb
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    wandb_name_and_id = f'BraTS23_UHVED_epoch{args.num_epochs}_lr{args.lr}_jobid{slurm_job_id}'
    if not args.debug:
        wandb.init(
            project="SegmentationMM",
            name=wandb_name_and_id,
            #entity="maxillo",
            id=wandb_name_and_id,
            resume="allow",
            config={
                "architecture": "U-HVED",
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "datapath": args.datapath,
            }
        )
    
    ##########setting models
    num_cls = 4
    model = U_HVED(num_classes=num_cls)
    print(model)
    model = torch.nn.DataParallel(model).cuda()

    ########## Setting learning scheduler and optimizer
    lr_scheduler = LR_Scheduler_polinomial(args.lr, args.num_epochs) #LR_Scheduler(args.lr) 
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer=optimizer)

    ########## Setting data
    train_file = 'datalist/train.txt'
    test_file = 'datalist/test15splits.csv'
    val_file = 'datalist/val15splits.csv'
    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls, train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    val_set = Brats_loadall_val_nii(transforms=args.test_transforms, root=args.datapath, num_cls=num_cls, val_file=val_file)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    val_loader = MultiEpochsDataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    iter_per_epoch = len(train_loader) #number of batches
    train_iter = iter(train_loader)
    val_Dice_best = -999999
    start_epoch = 0

    ##########Resume Training
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        val_Dice_best = checkpoint['val_Dice_best']
        optimizer.load_state_dict(checkpoint['optim_dict'])
        start_epoch = checkpoint['epoch'] + 1
        scheduler.load_state_dict(checkpoint['scheduler'])

    for epoch in range(start_epoch, args.num_epochs):
        #step_lr = lr_scheduler(optimizer, epoch)
        b = time.time()
        model.train()

        cross_loss_epoch = 0.0
        dice_loss_epoch = 0.0
        reconstruction_loss_epoch = 0.0
        kld_loss_epoch = 0.0
        loss_epoch = 0.0

        ########## training epoch
        for i in range(iter_per_epoch):
            #step_lr = lr_scheduler(optimizer, epoch*iter_per_epoch)             #current iter = epoch*iter_per_epoch
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]                                      #x=(B, M=4, 112, 112, 112), target (one-hot) = (B, C, 112, 112, 112), mask = (B, 4)
            x = x.cuda(non_blocking=True)                      
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            images = dict() 
            images['Flair'] = x[:,0:1,:,:,:]
            images['T1c'] = x[:,1:2,:,:,:]
            images['T1'] = x[:,2:3,:,:,:]
            images['T2'] = x[:,3:4,:,:,:]

            net_img, post_param = model(images, mask, is_inference=False)
            
            ###Loss compute
            ### segmentation loss
            loss_cross = softmax_loss(net_img['seg'], target, num_cls=num_cls)
            loss_dice = dice_loss(net_img['seg'], target, num_cls=num_cls)

            cross_loss_epoch += loss_cross
            dice_loss_epoch += loss_dice

            ### loss reconstruction
            rec_img = torch.zeros_like(x).cuda().float()
            rec_img[:,0:1,:,:,:] =  net_img['Flair'] - images['Flair']
            rec_img[:,1:2,:,:,:] =  net_img['T1c'] - images['T1c']
            rec_img[:,2:3,:,:,:] =  net_img['T1'] - images['T1']
            rec_img[:,3:4,:,:,:] =  net_img['T2'] - images['T2']
            loss_reconstruction = torch.mean(torch.square(rec_img))
            reconstruction_loss_epoch += loss_reconstruction

            ### KL divergence loss
            sum_inter_KLD = torch.zeros(1).cuda().float()
            sum_prior_KLD = torch.zeros(1).cuda().float()
            kld_loss = torch.zeros(1).cuda().float()
            nb_skip = len(post_param)
            for k in range(nb_skip):
                inter_KLD, prior_KLD = compute_KLD(post_param[k]['mu'], post_param[k]['logvar'], mask)
                sum_inter_KLD += inter_KLD
                sum_prior_KLD += prior_KLD
            kld_loss = 1 / nb_skip * sum_inter_KLD + 1 / nb_skip * sum_prior_KLD
            kld_loss_epoch += kld_loss

            ### total segmentation loss
            loss = loss_cross + loss_dice + 0.1 * kld_loss + 0.1 * loss_reconstruction
            loss_epoch += loss

            ### backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('kld_loss', kld_loss.item(), global_step=step)
            writer.add_scalar('reconstruction_loss', loss_reconstruction.item(), global_step=step)
            writer.add_scalar('cross_loss', loss_cross.item(), global_step=step)
            writer.add_scalar('dice_loss', loss_dice.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'kld_loss:{:.4f}, '.format(kld_loss.item())
            msg += 'reconstruction_loss:{:.4f},'.format(loss_reconstruction.item())
            msg += 'cross_loss:{:.4f}'.format(loss_cross.item())
            msg += 'dice_loss:{:.4f}'.format(loss_dice.item())
            logging.info(msg)

            #if not args.debug:
            #    wandb.log({"train/learning_rate": step_lr})

            #if args.debug:
            #    break

        logging.info('train time per epoch: {}'.format(time.time() - b))

        ########## log current epoch metrics and save current model 
        if not args.debug:
            wandb.log({
                "train/epoch": epoch,
                "train/loss": loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/loss_kld": kld_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/loss_reconstruction": reconstruction_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/loss_cross": cross_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/loss_dice": dice_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
            })

        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'val_Dice_best': val_Dice_best,
            'scheduler': scheduler.state_dict(),
            },
            file_name)
        
        ########## validation and test
        if (epoch+1) in val_check or args.debug:
            print('validate ...')
            with torch.no_grad():
                dice_score, seg_loss = test_softmax(
                    val_loader,
                    model,
                    dataname = args.dataname)
            scheduler.step(seg_loss)

            val_WT, val_TC, val_ET, val_ETpp = dice_score #validate(model, val_loader)
            logging.info('Validate epoch = {}, WT = {:.2}, TC = {:.2}, ET = {:.2}, ETpp = {:.2}, loss = {:.2}'.format(epoch, val_WT.item(), val_TC.item(), val_ET.item(), val_ETpp.item(), seg_loss.cpu().item()))
            val_dice = (val_ET + val_WT + val_TC)/3
            if not args.debug:
                wandb.log({
                    "val/epoch":epoch,
                    "val/val_ET_Dice": val_ET.item(),
                    "val/val_ETpp_Dice": val_ETpp.item(),
                    "val/val_WT_Dice": val_WT.item(),
                    "val/val_TC_Dice": val_TC.item(),
                    "val/val_Dice": val_dice.item(), 
                    "val/seg_loss": seg_loss.cpu().item(),       
                })
                
            if val_dice > val_Dice_best:
                val_Dice_best = val_dice.item()
                print('save best model ...')
                file_name = os.path.join(ckpts, 'best.pth')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'val_Dice_best': val_Dice_best,
                    'scheduler': scheduler.state_dict(),
                    },
                    file_name)
                
            print('testing ...')
            with torch.no_grad():
                dice_score, seg_loss = test_softmax(
                    test_loader,
                    model,
                    dataname = args.dataname)
            test_WT, test_TC, test_ET, test_ETpp = dice_score   
            logging.info('Testing epoch = {}, WT = {:.2}, TC = {:.2}, ET = {:.2}, ET_postpro = {:.2}'.format(epoch, test_WT.item(), test_TC.item(), test_ET.item(), test_ETpp.item()))
            test_dice = (test_ET + test_WT + test_TC)/3
            if not args.debug:
                wandb.log({
                    "test/epoch":epoch,
                    "test/test_WT_Dice": test_WT.item(),
                    "test/test_TC_Dice": test_TC.item(),
                    "test/test_ET_Dice": test_ET.item(),
                    "test/test_ETpp": test_ETpp.item(),
                    "test/test_Dice": test_dice.item(),  
                    "test/seg_loss": seg_loss.cpu().item(),   
                })

            model.train()
            #model.module.is_training=True

    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

if __name__ == '__main__':
    main()
