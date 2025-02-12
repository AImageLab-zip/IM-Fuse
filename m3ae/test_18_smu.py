import argparse
import os
import pathlib
import time
import csv

from medpy.metric import binary # for hd

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn as nn
import yaml
from monai.data import decollate_batch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from dataset.brats import get_datasets_train_rf_withvalid,get_datasets_train_rf_withtest, get_datasets23_train_rf_withvalidtest
from loss import EDiceLoss
from loss.dice import EDiceLoss_Val
from utils import AverageMeter, ProgressMeter, save_checkpoint, reload_ckpt_bis, \
    count_parameters, save_metrics, save_args_1, inference, post_trans, dice_metric, \
    dice_metric_batch, reload_ckpt
from model.Unet import Unet_missing

from torch.cuda.amp import autocast as autocast

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.cuda.set_device(0)

masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]

mask_name = ['t2', 't1c', 't1', 'flair',
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']

parser = argparse.ArgumentParser(description='VTUNET BRATS 2021 Training')
# DO not use data_aug argument this argument!!
parser.add_argument('--modal_list', nargs='+', help='<Required> Set flag')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--mdp', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('--devices', default='0', type=str, help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--checkpoint', default='/data5/lh/brats/runs/sd_new6/model_1model_best_449.pth.tar', type=str, help='Set the CUDA_VISIBLE_DEVICES env var from this string')


parser.add_argument('--exp_name', default='baseline3_real', type=str, help='exp name')

parser.add_argument('--val', default=20, type=int, help="how often to perform validation step")
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default="configs/vt_unet_costum.yaml", metavar="FILE",
                    help='path to config file', )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')


parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', default=True, type=bool, help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--mae_imp', default=True, type=bool, help='resume from checkpoint')

parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

device = torch.device("cuda:0")


def main(args):
    # setup
    ngpus = torch.cuda.device_count()
    print(f"Working with {ngpus} GPUs")
    print(args.checkpoint)
    args.checkpoint = args.checkpoint
    args.save_folder_1 = pathlib.Path(f"./runs/{args.exp_name}/model_1")
    args.save_folder_1.mkdir(parents=True, exist_ok=True)
    args.seg_folder_1 = args.save_folder_1 / "segs"
    args.seg_folder_1.mkdir(parents=True, exist_ok=True)
    args.save_folder_1 = args.save_folder_1.resolve()
    save_args_1(args)
    
    t_writer_1 = SummaryWriter(str(args.save_folder_1))
    args.checkpoint_folder = pathlib.Path(f"./runs/{args.exp_name}/model_1")

    print(args)
    
    if args.modal_list:
        args.modal_list = [int(l) for l in args.modal_list]
    else:
        args.modal_list = []
    
    # Create model
    
    model_1 = Unet_missing(input_shape = [128,128,128], out_channels=3, mdp=3, init_channels = 16,  pre_train = False, mask_modal = args.modal_list, patch_shape = 128)

    
    temp_model = model_1
    temp_model.eval()

    model_1 = nn.DataParallel(model_1)
    if args.resume:
        ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        model_1.load_state_dict(ck["state_dict"], strict=False)
    
    print(f"total number of trainable parameters {count_parameters(model_1)}")

    model_1 = model_1.cuda()

    model_file = args.save_folder_1 / "model.txt"
    with model_file.open("w") as f:
        print(model_1, file=f)

    #full_train_dataset, l_val_dataset, l_test_dataset, _,_labeled_name_list = get_datasets23_train_rf_withvalidtest(args.seed, fold_number=args.fold, part = args.part, all_data = args.all_data, patch_shape = args.patch_shape)
    full_train_dataset, l_val_dataset, l_test_dataset, _,_labeled_name_list = get_datasets23_train_rf_withvalidtest(args.seed, fold_number=args.fold, patch_shape = 128)

    test_loader = torch.utils.data.DataLoader(l_test_dataset, batch_size=1, shuffle=False,
                                              pin_memory=True, num_workers=args.workers)

    criterian_val = EDiceLoss_Val().cuda()
    metric = criterian_val.metric
    all_dice = []
    output_path = args.save_folder_1 / "test_final.txt"

    for m in masks:
        test_loss, test_metrics, dice_mean = eval_step(test_loader, model_1, metric, t_writer_1, mask = m, save_folder = output_path)
        all_dice.append(dice_mean)

    dice_avg = np.array(all_dice).mean(axis=0)

    with open(output_path, 'a') as file:
        file.write(
            "Overall Averages: WT = {:.4f}, TC = {:.4f}, ET = {:.4f}\n".format(
                dice_avg[2], dice_avg[1], dice_avg[0]))

    '''criterion = EDiceLoss().cuda()
    criterian_val = EDiceLoss_Val().cuda()
    metric = criterian_val.metric
    print(metric)
    params = model_1.parameters()
    
    limage = []
    ori_para = []
    
    for pname, p in model_1.named_parameters():
        if pname.endswith("limage"):
            limage.append(p)
        else:
            ori_para.append(p)

    optimizer = torch.optim.Adam(ori_para, lr=args.lr, weight_decay=args.weight_decay)
    
    if args.val != 0:
        full_train_dataset, l_val_dataset, _, _ = get_datasets_train_rf_withvalid(args.seed, fold_number=args.fold)
    else:
        full_train_dataset, l_val_dataset, _, _ = get_datasets_train_rf_withtest(args.seed, fold_number=args.fold)
    train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(l_val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)
    #bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=args.workers)

    print("Train dataset number of batch:", len(train_loader))
    print("Val dataset number of batch:", len(val_loader))
    #print("Bench Test dataset number of batch:", len(bench_loader))

    # Actual Train loop
    patients_perf = []
        
    # do_epoch for one epoch
    ts = time.perf_counter()

    # Setup
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_ = AverageMeter('Loss', ':.4e')
    epoch = 0
    model_1.train()
    
    mode = "train" if model_1.training else "val"
    batch_per_epoch = len(train_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses_],
        prefix=f"{mode} Epoch: [{epoch}]")

    end = time.perf_counter()
    metrics = []

    
    validation_loss_1, validation_dice = step(val_loader, model_1, temp_model, criterian_val, metric, epoch, t_writer_1,
                                                  save_folder=args.save_folder_1,
                                                  patients_perf=patients_perf, args=args)

    t_writer_1.add_scalar(f"SummaryLoss", validation_loss_1, epoch)
    t_writer_1.add_scalar(f"SummaryDice", validation_dice, epoch)

    print(validation_dice)
    
    ts = time.perf_counter()
            '''


def eval_step(data_loader, model, metric, writer, mask, save_folder=None):

    # Setup
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses],
        prefix=f"Evaluation step")

    end = time.perf_counter()
    metrics = []

    odice_metric = []
    hd_metric = []
    hd95_metric = []

    model.module.mask_modal = [i for i, value in enumerate(mask) if value == False]

    for i, val_data in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)
        #patient_id = val_data["patient_id"]

        model.eval()
        with torch.no_grad():
            val_inputs, val_labels = (
                val_data["image"].cuda(),
                val_data["label"].cuda(),
            )

            val_outputs = inference(val_inputs, model)
            val_outputs_1 = [post_trans(i) for i in decollate_batch(val_outputs)]

            segs = val_outputs
            targets = val_labels
            #loss_ = criterion(segs, targets)
            dice_metric(y_pred=val_outputs_1, y=val_labels)

        '''if patients_perf is not None:
            patients_perf.append(
                dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_.item())
            )'''

        metric_ = metric(segs, targets)
        metrics.extend(metric_)

        hd = []
        hd95 = []
        dice = []
        for l in range(segs.shape[1]):
            if targets[0, l].cpu().numpy().sum() == 0:
                hd.append(1)
                hd95.append(0)
                dice.append(metric_[0][l].cpu().numpy())
                print((segs[0, l].cpu().numpy() > 0.5).sum())

                continue
            if (segs[0, l].cpu().numpy() > 0.5).sum() == 0:
                hd.append(0)
                hd95.append(0)
                dice.append(metric_[0][l].cpu().numpy())
                continue

            hd.append(binary.hd(segs[0, l].cpu().numpy() > 0.5, targets[0, l].cpu().numpy() > 0.5, voxelspacing=None))
            # hd95
            hd95.append(
                binary.hd95(segs[0, l].cpu().numpy() > 0.5, targets[0, l].cpu().numpy() > 0.5, voxelspacing=None))

            dice.append(metric_[0][l].cpu().numpy())
        hd_metric.append(hd)
        hd95_metric.append(hd95)
        odice_metric.append(dice)

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        # Display progress
        progress.display(i)

    #save_metrics(epoch, metrics, writer, epoch, False, save_folder)
    #writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)

    #dice_values = dice_metric.aggregate().item()
    dice_metric.reset()
    dice_metric_batch.reset()

    metricss = list(zip(*metrics))
    metrics = [np.nanmean(torch.tensor(dice, device="cpu").numpy()) for dice in metricss]
    #stds = [np.nanstd(torch.tensor(dice, device="cpu").numpy()) for dice in metricss]

    #hd_metric = [np.nanmean(l) for l in zip(*hd_metric)]
    #hd95_metric = [np.nanmean(l) for l in zip(*hd95_metric)]
    #dice_std = [np.nanstd(l) for l in zip(*odice_metric)]
    dice_mean = [np.nanmean(l) for l in zip(*odice_metric)]

    # print([l for l in zip(*odice_metric)])
    '''print("hd: ", hd_metric)
    print("hd_95: ", hd95_metric)
    print("dice_std:", dice_std)'''



    '''file = open("smu2.csv", "a+")
    csv_writer = csv.writer(file)
    csv_writer.writerow([*hd_metric, *hd95_metric, *dice_std, *dice_mean])
    file.close()'''

    with save_folder.open("a") as file:
        file.write(
            'Performance missing scenario = {}, WT = {:.4f}, TC = {:.4f}, ET = {:.4f}\n'.format(mask,
                                                                                                               dice_mean[2].item(),
                                                                                                               dice_mean[1].item(),
                                                                                                               dice_mean[0].item()))

    return losses.avg, np.nanmean(metrics), dice_mean  # , metrics

def step(data_loader, model, model_temp, criterion: EDiceLoss_Val, metric, epoch, writer, save_folder=None, patients_perf=None, args = None):
    # Setup
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    mode = "val"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses],
        prefix=f"{mode} Epoch: [{epoch}]")

    end = time.perf_counter()
    metrics = []
    
    hd_metric = []
    hd95_metric = []
    odice_metric = []

    for i, val_data in enumerate(data_loader):

        # measure data loading time
        data_time.update(time.perf_counter() - end)

        patient_id = val_data["patient_id"]
        
        
        
        model.eval()
        with torch.no_grad():
            if not args.mae_imp or len(args.modal_list) == 0:
                val_inputs, val_labels = (
                    val_data["image"].cuda(),
                    val_data["label"].cuda(),
                )
                
                
                val_outputs = inference(val_inputs, model)
                
            else:
                val_inputs, val_labels = (
                    val_data["image"].cuda(),
                    val_data["label"].cuda(),
                )
                syn_result = inference(val_inputs, model_temp)
                syn_result = syn_result.cuda()
                val_outputs = inference(syn_result, model)
            
            val_outputs_1 = [post_trans(i) for i in decollate_batch(val_outputs)]
            
            segs = val_outputs
            targets = val_labels
            loss_ = criterion(segs, targets)
            dice_metric(y_pred=val_outputs_1, y=val_labels)

        if patients_perf is not None:
            patients_perf.append(
                dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_.item())
            )

        writer.add_scalar(f"Loss/{mode}{''}",
                          loss_.item(),
                          global_step=batch_per_epoch * epoch + i)


        metric_ = metric(segs, targets)
        metrics.extend(metric_)
                
        hd = []
        hd95 = []
        dice = []
        for l in range(segs.shape[1]):
            if targets[0,l].cpu().numpy().sum() == 0:
                hd.append(1)
                hd95.append(0)
                dice.append(metric_[0][l].cpu().numpy())
                print((segs[0,l].cpu().numpy() > 0.5).sum())
                
                continue
            if (segs[0,l].cpu().numpy() > 0.5).sum() == 0:
                hd.append(0)
                hd95.append(0)
                dice.append(metric_[0][l].cpu().numpy())
                continue

            hd.append(binary.hd(segs[0,l].cpu().numpy() > 0.5, targets[0,l].cpu().numpy() > 0.5, voxelspacing=None))
            #hd95
            hd95.append(binary.hd95(segs[0,l].cpu().numpy() > 0.5, targets[0,l].cpu().numpy() > 0.5, voxelspacing=None))
            
            dice.append(metric_[0][l].cpu().numpy())
        hd_metric.append(hd)
        hd95_metric.append(hd95)
        odice_metric.append(dice)
        
        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        # Display progress
        progress.display(i)

    save_metrics(epoch, metrics, writer, epoch, False, save_folder)
    writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)

    dice_values = dice_metric.aggregate().item()
    dice_metric.reset()
    dice_metric_batch.reset()
    
    metricss = list(zip(*metrics))
    metrics = [np.nanmean(torch.tensor(dice, device="cpu").numpy()) for dice in metricss]
    stds = [np.nanstd(torch.tensor(dice, device="cpu").numpy()) for dice in metricss]
    # print(stds)
    labels = ("ET", "TC", "WT")
    #metrics = {key: value for key, value in zip(labels, metrics)}
    
    hd_metric = [np.nanmean(l) for l in zip(*hd_metric)]
    hd95_metric = [np.nanmean(l) for l in zip(*hd95_metric)]
    dice_std = [np.nanstd(l) for l in zip(*odice_metric)]
    dice_mean = [np.nanmean(l) for l in zip(*odice_metric)]
    
    # print([l for l in zip(*odice_metric)])
    print("hd: ", hd_metric)
    print("hd_95: ", hd95_metric)
    print("dice_std:", dice_std)
    
    file = open("smu2.csv", "a+")
    csv_writer = csv.writer(file)
    csv_writer.writerow([*hd_metric, *hd95_metric, *dice_std, *dice_mean])
    file.close()
    
    return losses.avg, np.nanmean(metrics)#, metrics


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
