import argparse
import sys
sys.path.append("..")

import torch
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn

import os
import os.path as osp
from DualNet import DualNet
from BraTSDataSet import BraTSDataSet, BraTSValDataSet, my_collate
import timeit
import loss_Dual as loss
from engine import Engine
from math import ceil
from pathlib import Path
from DualNet import conv3x3x3
import wandb
import traceback
from tqdm import tqdm
from matplotlib import pyplot as plt
start = timeit.default_timer()
kd_wt = 0.1

def log_wandb_slice(images, masks):
    """
    images: torch.Size([B, 4, 80, 160, 160])
    masks:  torch.Size([B, 3, 80, 160, 160])
    """

    # image slice: [0, 0, 40]
    img_slice = images[0, 0, 40].detach().cpu().numpy()

    # segmentation: reduce channel dim -> binary mask
    # mask != 0 over channel dimension
    seg_binary = (masks[0] != 0).any(dim=0)  # (80, 160, 160)
    print(masks.sum())
    seg_slice = seg_binary[40].detach().cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img_slice, cmap="gray")
    ax.imshow(seg_slice, cmap="Reds", alpha=0.4)
    ax.axis("off")

    wandb.log(
        {"train/axial_slice": wandb.Image(fig)}
    )

    plt.close(fig)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    """
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Shared-Specific model for 3D Medical Image Segmentation.")

    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--train-list", type=str, default=Path(__file__).parent / 'datalist' / 'train.csv',required=False)
    parser.add_argument("--val-list", type=str, default=Path(__file__).parent / 'datalist' / 'val.csv',required=False)
    parser.add_argument("--checkpoint-path", type=str, default='snapshots/example/')
    parser.add_argument("--reload-path", type=str, default='snapshots/example/last.pth')
    parser.add_argument("--reload-from-checkpoint", type=str2bool, default=False)
    parser.add_argument("--input-size", type=str, default='80,160,160',required=False)
    parser.add_argument("--batch-size", type=int, default=2,required=False)
    parser.add_argument("--num-gpus", type=int, default=1,required=False)
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=int(80000*3.837719298),required=False)
    parser.add_argument("--start-iters", type=int, default=0)
    parser.add_argument("--val-pred-every", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--weight-std", type=str2bool, default=True,required=False)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--is-training", action="store_true")
    parser.add_argument("--random-mirror", type=str2bool, default=True,required=False)
    parser.add_argument("--random-scale", type=str2bool, default=True,required=False)
    parser.add_argument("--random-seed", type=int, default=999)
    parser.add_argument("--wandb-project-name",type=str,default=None)
    parser.add_argument("--norm-cfg", type=str, default='IN')  # normalization
    parser.add_argument("--activation-cfg", type=str, default='LeakyReLU')  # activation
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--mode", type=str, default='0,1,2,3')
    parser.add_argument("--teachers", type=str, default='0')
    parser.add_argument("--restart",action='store_true')

    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_steps, power):
    lr = lr_poly(lr, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def dice_score(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2*num / den

    return dice.mean()


def compute_dice_score(preds, labels):

    preds = F.sigmoid(preds)

    pred_ET = preds[:, 0, :, :, :]
    pred_WT = preds[:, 1, :, :, :]
    pred_TC = preds[:, 2, :, :, :]
    label_ET = labels[:, 0, :, :, :]
    label_WT = labels[:, 1, :, :, :]
    label_TC = labels[:, 2, :, :, :]
    dice_ET = dice_score(pred_ET, label_ET).cpu().data.numpy()
    dice_WT = dice_score(pred_WT, label_WT).cpu().data.numpy()
    dice_TC = dice_score(pred_TC, label_TC).cpu().data.numpy()
    return dice_ET, dice_WT, dice_TC


def predict_sliding(args, net, imagelist, tile_size, classes, mode):
    image, image_res = imagelist
    image_size = image.shape
    overlap = 1 / 3

    strideHW = ceil(tile_size[1] * (1 - overlap))
    strideD = ceil(tile_size[0] * (1 - overlap))
    tile_deps = int(ceil((image_size[2] - tile_size[0]) / strideD) + 1)
    tile_rows = int(ceil((image_size[3] - tile_size[1]) / strideHW) + 1)
    tile_cols = int(ceil((image_size[4] - tile_size[2]) / strideHW) + 1)
    full_probs = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)
    count_predictions = np.zeros((image_size[0], classes, image_size[2], image_size[3], image_size[4])).astype(np.float32)
    full_probs = torch.from_numpy(full_probs).cuda()
    count_predictions = torch.from_numpy(count_predictions).cuda()

    for dep in range(tile_deps):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d1 = int(dep * strideD)
                x1 = int(col * strideHW)
                y1 = int(row * strideHW)
                d2 = min(d1 + tile_size[0], image_size[2])
                x2 = min(x1 + tile_size[2], image_size[4])
                y2 = min(y1 + tile_size[1], image_size[3])
                d1 = max(int(d2 - tile_size[0]), 0)
                x1 = max(int(x2 - tile_size[2]), 0)
                y1 = max(int(y2 - tile_size[1]), 0)

                img = image[:, :, d1:d2, y1:y2, x1:x2]
                # img_res = image_res[:, :, d1:d2, y1:y2, x1:x2]

                prediction, _, _ = net(img, val=True, mode=mode)

                count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, :, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    return full_probs


def validate(args, input_size, model, ValLoader, num_classes):
    # start to validate
    val_ET = [0, 0, 0, 0]
    val_WT = [0, 0, 0, 0]
    val_TC = [0, 0, 0, 0]

    # return 2, 0, 2

    for index, batch in enumerate(ValLoader):
        # print('validation %d processd'%(index))
        image, image_res, label, size, name, affine = batch
        image = image.cuda()
        image_res = image_res.cuda()
        label = label.cuda()
        with torch.no_grad():
            for i_mode in [0, 1, 2, 3]:
                pred = predict_sliding(args, model, [image, image_res], input_size, num_classes, mode=str(i_mode))
                dice_ET, dice_WT, dice_TC = compute_dice_score(pred, label)
                val_ET[i_mode] += dice_ET
                val_WT[i_mode] += dice_WT
                val_TC[i_mode] += dice_TC

    print('Val_ET_Dice:', val_ET)
    print('Val_WT_Dice:', val_WT)
    print('Val_TC_Dice:', val_TC)
    if args.wandb_project_name is not None:
        wandb.log({
            'val/et_dice':val_ET,
            'val/wt_dice':val_WT,
            'val/tc_dice':val_TC
        })
    return val_ET.index(max(val_ET)), val_WT.index(max(val_WT)), val_TC.index(max(val_TC))


def main():
    try:
        """Create the ConResNet model and then start the training."""
        parser = get_arguments()
        
        # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        with Engine(custom_parser=parser) as engine:
            args = parser.parse_args()
            args.learning_rate = args.learning_rate * np.sqrt(args.batch_size/2)
            if args.wandb_project_name is not None:
                wandb.init(project=args.wandb_project_name,name='training')
            args.datapath = Path(args.datapath)
            if args.num_gpus > 1:
                torch.cuda.set_device(args.local_rank)

            d, h, w = map(int, args.input_size.split(','))
            input_size = (d, h, w)

            cudnn.benchmark = True
            seed = args.random_seed
            if engine.distributed:
                seed = args.local_rank
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            model = DualNet(args=args, norm_cfg=args.norm_cfg, activation_cfg=args.activation_cfg,
                            num_classes=args.num_classes, weight_std=args.weight_std, self_att=False, cross_att=False)
            model.train()
            device = torch.device('cuda:{}'.format(args.local_rank))
            model.to(device)

            optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)

            if args.num_gpus > 1:
                model = engine.data_parallel(model)

            # load checkpoint...
            if args.reload_from_checkpoint:
                print('loading from checkpoint: {}'.format(args.reload_path))
                if os.path.exists(args.reload_path):
                    checkpoint = torch.load(args.reload_path,weights_only=False)
                    model.load_state_dict(checkpoint['model'])
                    if not args.restart: # Load optimizer and start_iters only if required
                        optimizer = checkpoint['optimizer'] 
                        args.start_iters = checkpoint['iter']
                    print("Loaded model trained for", args.start_iters, "iters")
                else:   
                    print('File not exists in the reload path: {}'.format(args.reload_path))
                    exit(0)

            loss_D = loss.DiceLoss4BraTS().to(device)
            loss_BCE = loss.BCELoss4BraTS().to(device)

            print('current mode:', args.mode)

            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            train_dataset = BraTSDataSet(args.datapath, Path(args.train_list), max_iters=args.num_steps * 2/args.batch_size, crop_size=input_size,
                            scale=args.random_scale, mirror=args.random_mirror)
            val_dataset = BraTSValDataSet(args.datapath, args.val_list)
            trainloader, train_sampler = engine.get_train_loader(train_dataset)
            valloader, val_sampler = engine.get_test_loader(val_dataset)

            val_Dice_best = -999999
            teachers = args.teachers
            for i_iter, batch in tqdm(enumerate(trainloader),total=len(trainloader)):
                i_iter += args.start_iters
                images = batch['image'].squeeze(1).cuda()
                labels = batch['label'].squeeze(1).cuda()
                optimizer.zero_grad()
                lr = adjust_learning_rate(optimizer, i_iter, args.learning_rate, len(trainloader), args.power)
                preds, mode_split, totKDLoss = model(images, mode=args.mode, teachers=teachers)
                preds_seg = preds
                
                term_seg_Dice = loss_D.forward(preds_seg, labels)
                term_seg_BCE = loss_BCE.forward(preds_seg, labels)

                term_all = term_seg_Dice + term_seg_BCE + kd_wt * totKDLoss

                term_all.backward()
                optimizer.step()
                if i_iter % 5 == 0 and (args.local_rank == 0) and args.wandb_project_name is not None:
                    wandb.log({
                        'train/loss':term_all.detach().cpu().item(),
                        'train/lr':lr,
                        'train/iter':i_iter
                    })
                    #preds_for_logging = torch.round(F.sigmoid(preds))
                    #log_wandb_slice(images,preds_for_logging)
                '''print('iter = {} of {} completed, lr = {:.4}, seg_loss = {:.4}, kd_loss = {:.4}'
                    .format(i_iter, args.num_steps, lr, (term_seg_Dice+term_seg_BCE).cpu().data.numpy(), totKDLoss.cpu().data.numpy()))
    '''
                if i_iter >= len(trainloader) - 1 and (args.local_rank == 0):
                    print('save last model ...')
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer,
                        'iter': i_iter
                    }
                    torch.save(checkpoint, osp.join(args.checkpoint_path, 'final.pth'))
                    break

                if i_iter % args.val_pred_every == args.val_pred_every - 1 and i_iter != 0 and (args.local_rank == 0):
                    print('save model ...')
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer,
                        'iter': i_iter
                    }
                    # torch.save(checkpoint, osp.join(args.checkpoint_path, 'iter_' + str(i_iter) + '.pth'))
                    torch.save(checkpoint, osp.join(args.checkpoint_path, 'last.pth'))

                # val and identify the best modality for each tumor
                '''if not args.train_only and (i_iter +1)% args.val_pred_every == 0:
                    print('validate ...')
                    #teacher_modes = validate(args, input_size, model, valloader, args.num_classes)
                    # teachers = [max(teacher_modes, key=teacher_modes.count)]  # single teacher
                    teachers = list(set(teacher_modes))  # multi-teacher
                    teachers = ",".join(map(str, teachers))
                    print('teachers:', teachers)'''

        end = timeit.default_timer()
        print(end - start, 'seconds')
    except:
        train_dataset.close()
        tb = traceback.format_exc()
        print(tb)       


if __name__ == '__main__':
    main()
