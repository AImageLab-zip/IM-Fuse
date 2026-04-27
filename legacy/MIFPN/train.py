#coding=utf-8
import argparse
import logging
import os
import time
from data.data_utils import init_fn
from data.datasets_nii import (Brats_loadall_nii, Brats_loadall_test_nii,
                               Brats_loadall_val_nii)
from data.transforms import *
from models import PNT
from predict import AverageMeter, test_softmax
from torch.utils.tensorboard import SummaryWriter
from utils import criterions
from utils.lr_scheduler import LR_Scheduler, MultiEpochsDataLoader
from utils.parser import setup
from utils.random_seed import setup_seed
import wandb

parser = argparse.ArgumentParser()


parser.add_argument('--model', default='PNT', type=str)
parser.add_argument('-batch_size', '--batch_size', default=2, type=int, help='Batch size')
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--start_epochs', default=777, type=int)
parser.add_argument('--iter_per_epoch', default=200, type=int)
parser.add_argument('--dataname', default='BRATS2020', type=str)
parser.add_argument('--datapath', default='/home/diaoyueqin/diaoyueqin/BraTS/BRATS2020_Training_none_npy', type=str)
parser.add_argument('--savepath', default='/home/diaoyueqin/dyq/M2FTrans_v1/output', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=999, type=int)
#parser.add_argument('--needvalid', default=False, type=bool)
#parser.add_argument('--csvpath', default='/home/diaoyueqin/dyq/M2FTrans_v1/output', type=str)
parser.add_argument('--debug', action='store_true', default=False)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

#csvpath = args.csvpath
#os.makedirs(csvpath, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks_test))
mask_name = ['t2', 't1c', 't1', 'flair',
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_torch.int())

val_check = [50, 70, 90, 110, 130, 150, 200, 300, 400, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 800, 825, 850, 900, 910, 920, 930, 940, 950, 955, 960, 965, 970, 975, 980, 985, 990, 995, 1000] 
print(f"Validation checks: {val_check}")

masks_valid = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]

def main():
    ########## setting seed
    setup_seed(args.seed)
    
    ########## print args
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True)

    ########## init wandb
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    wandb_name_and_id = f'{args.dataname}_MIFPN_epoch{args.num_epochs}_iter{args.iter_per_epoch}_jobid{slurm_job_id}'
    if not args.debug:
        wandb.init(
            project="SegmentationMM",
            name=wandb_name_and_id,
            #entity="maxillo",
            id=wandb_name_and_id,
            resume="allow",
            config={
                "architecture": "MIFPN",
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "iter_per_epoch": args.iter_per_epoch,
                "num_epochs": args.num_epochs,
                "datapath": args.datapath,
                "region_fusion_start_epoch": args.region_fusion_start_epoch,
            }
        )
    ########## setting models
    if args.dataname in ['BRATS2023', 'BRATS2021', 'BRATS2020', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print ('dataset is error')
        exit(0)

    model = PNT.Model(num_cls=num_cls)
    model = torch.nn.DataParallel(model).cuda()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameter_count:{count_parameters(model)}")
    # input_tensor = torch.randn(1,4, 80, 80, 80)
    # input_tensor=(input_tensor,mask)
    # flops = FlopCountAnalysis(model, input_tensor)
    # print(f"Model FLOPs: {flops.total()}")

    ########## Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs, warmup=args.region_fusion_start_epoch, mode='warmuppoly')
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.AdamW(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    
    ########## Setting data
    if args.dataname in ['BRATS2023', 'BRATS2020', 'BRATS2015']:
        train_file = 'datalist/train.txt'
        test_file = 'datalist/test15splits2.csv'
        val_file = 'datalist/val15splits.csv'
        #test_file = 'datalist/test.txt'
        #val_file = 'datalist/val.txt'
    elif args.dataname == 'BRATS2018':
        #### BRATS2018 contains three splits (1,2,3)
        test_file = 'datalist/Brats18_test15splits.csv'
        val_file = 'datalist/Brats18_val15splits.csv'
        train_file = 'datalist/Brats18_train3.txt'
    
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

    ########## Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    # iter_per_epoch = args.iter_per_epoch
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

    for epoch in range(start_epoch, args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        b = time.time()
        model.train()
        model.module.is_training = True

        prm_cross_loss_epoch = 0.0
        prm_dice_loss_epoch = 0.0
        fuse_cross_loss_epoch = 0.0
        fuse_dice_loss_epoch = 0.0
        sep_cross_loss_epoch = 0.0
        sep_dice_loss_epoch = 0.0
        loss_epoch = 0.0
        kl_loss_epoch = 0.0

        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ### Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)

            x, target, mask = data[:3]
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            batchsize = mask.size(0)
            mask = mask[0].repeat(batchsize, 1)  ## to be test
            mask = mask.cuda(non_blocking=True)

            fuse_pred, sep_preds, prm_preds, kl= model(x, mask)

            ### Loss compute
            ### fuse modality segmentation loss
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            fuse_cross_loss_epoch += fuse_cross_loss
            fuse_dice_loss_epoch += fuse_dice_loss

            ### separated modality segmentation loss
            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_cross_loss + sep_dice_loss

            sep_cross_loss_epoch += sep_cross_loss
            sep_dice_loss_epoch += sep_dice_loss

            kl_loss = criterions.softmax_kl_loss(kl)
            kl_loss_epoch += kl_loss

            ### pyramid segmentation loss
            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            prm_loss = prm_cross_loss + prm_dice_loss

            prm_cross_loss_epoch += prm_cross_loss
            prm_dice_loss_epoch += prm_dice_loss

            if epoch < args.region_fusion_start_epoch:
                loss = fuse_loss * 0.0 + sep_loss + prm_loss * 0.0
            else:
                # loss = fuse_loss+ sep_loss 
                loss = fuse_loss + sep_loss + prm_loss + kl_loss
            
            loss_epoch += loss

            ### backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)
            writer.add_scalar('kl_loss', kl_loss.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())
            msg += 'kl_loss:{:.4f}'.format(kl_loss.item())
            logging.info(msg)

        b_train = time.time()

        logging.info('train time per epoch: {}'.format(b_train - b))

        ########## log current epoch metrics and save current model 
        if not args.debug:
            wandb.log({
                "train/epoch": epoch,
                "train/loss": loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/fusecross": fuse_cross_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/fusedice": fuse_dice_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/sepcross": sep_cross_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/sepdice": sep_dice_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/prmcross": prm_cross_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/prmdice": prm_dice_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/kl_loss": kl_loss_epoch.cpu().detach().item() / iter_per_epoch,
                "train/learning_rate": step_lr,
            })

        ######### model save
        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'val_Dice_best': val_Dice_best,
            },
            file_name)
        
        ########## validation and test
        if (epoch + 1) in val_check:
            print('validate ...')
            with torch.no_grad():
                dice_score, seg_loss = test_softmax(
                    val_loader,
                    model,
                    dataname = args.dataname)
        
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
                    },
                    file_name)
                
            print('testing ...')
            test_score = AverageMeter()
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
            model.module.is_training=True

    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)


if __name__ == '__main__':
    main()
