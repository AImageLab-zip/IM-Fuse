import argparse
import sys
sys.path.append("..")
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from utils.random_seed import setup_seed
import os
from DualNet_SS import DualNet_SS as DualNet
from BraTSDataSet import BraTSValDataSet
import loss_Dual as loss
from math import ceil

alpha = 0.1  # shared domain loss weight
beta = 0.02  # specific domain loss weight
calc_flops = False

def get_arguments():
    """
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Shared-Specific model for 3D Medical Image Segmentation.")

    parser.add_argument("--data_dir", type=str, default='./datalist/')
    parser.add_argument("--val_list", type=str, default='BraTS23/BraTS23_val15splits.csv')
    parser.add_argument("--test_list", type=str, default='BraTS23/BraTS23_test15splits.csv')
    parser.add_argument("--reload_path", type=str, default='/work/grana_neuro/missing_modalities/ShaSpec/snapshots/BraTS23_ShaSpec_[80,160,160]_SGD_b1_lr-2_alpha.1_beta.02_training3/final.pth') 
    parser.add_argument("--input_size", type=str, default='80,160,160')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--weight_std", action='store_true', default=True)
    parser.add_argument("--random_seed", type=int, default=999)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--norm_cfg", type=str, default='IN')  # normalization
    parser.add_argument("--activation_cfg", type=str, default='LeakyReLU')  # activation
    parser.add_argument("--mode", type=str, default='random')

    return parser

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


def predict_sliding(args, net, imagelist, mask_indices, tile_size, classes):
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
                img_res = image_res[:, :, d1:d2, y1:y2, x1:x2]

                prediction, _, _, _ = net(img, mask_indices, val=True, mode=args.mode)

                count_predictions[:, :, d1:d2, y1:y2, x1:x2] += 1
                full_probs[:, :, d1:d2, y1:y2, x1:x2] += prediction

    full_probs /= count_predictions
    return full_probs


def validate(args, input_size, model, ValLoader, num_classes, loss_D, loss_BCE):
    # start to validate
    val_ET = 0.0
    val_WT = 0.0
    val_TC = 0.0
    loss_seg = 0.0

    for index, batch in enumerate(ValLoader):
        image, image_res, label, mask, mask_indices, size, name, affine = batch
        image = image.cuda()
        image_res = image_res.cuda()
        label = label.cuda()
        mask_indices = mask_indices[0]
        #mask = mask.cuda()

        with torch.no_grad():
            pred = predict_sliding(args, model, [image, image_res], mask_indices, input_size, num_classes)
            
            term_seg_Dice = loss_D.forward(pred, label)
            term_seg_BCE = loss_BCE.forward(pred, label)
            term_seg = term_seg_Dice + term_seg_BCE

            dice_ET, dice_WT, dice_TC = compute_dice_score(pred, label)

            val_ET += dice_ET
            val_WT += dice_WT
            val_TC += dice_TC
            loss_seg += term_seg

    return val_ET/(index+1), val_WT/(index+1), val_TC/(index+1), loss_seg/(index+1)


def main():
    """Create the ConResNet model and then start the training."""
    parser = get_arguments()
    print(parser)
    print(torch.cuda.is_available())


    ########## setting 
    args = parser.parse_args()
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True)
    print('current mode:', args.mode)
    d, h, w = map(int, args.input_size.split(','))
    input_size = (d, h, w)

    ########## setting seed
    setup_seed(args.random_seed)
    
    ########## setting model
    model = DualNet(args=args, norm_cfg=args.norm_cfg, activation_cfg=args.activation_cfg,
                    num_classes=args.num_classes, weight_std=args.weight_std, self_att=False, cross_att=False)
    device = torch.device('cuda:{}'.format(args.local_rank))
    model.to(device)
    model.eval()

    loss_D = loss.DiceLoss4BraTS().to(device)
    loss_BCE = loss.BCELoss4BraTS().to(device)

    ########## setting data
    valloader = torch.utils.data.DataLoader(BraTSValDataSet(args.data_dir, args.val_list),
                                    batch_size=1,
                                    num_workers=args.num_workers,
                                    drop_last=False,
                                    shuffle=False,
                                    pin_memory=True,
                                    sampler=None)    
    
    testloader = torch.utils.data.DataLoader(BraTSValDataSet(args.data_dir, args.test_list),
                                    batch_size=1,
                                    num_workers=args.num_workers,
                                    drop_last=False,
                                    shuffle=False,
                                    pin_memory=True,
                                    sampler=None)

    ########## load checkpoint...
    print('loading from checkpoint: {}'.format(args.reload_path))
    if os.path.exists(args.reload_path):
        checkpoint = torch.load(args.reload_path)
        model = checkpoint['model']
        args.start_iters = checkpoint['iter']
        print("Loaded model trained for", args.start_iters, "iters")
    else:
        print('File not exists in the reload path: {}'.format(args.reload_path))
        exit(0)

    print('validate ...')
    val_ET, val_WT, val_TC, seg_loss = validate(args, input_size, model, valloader, args.num_classes, loss_D, loss_BCE)
    print('Validate ET = {:.2}, WT = {:.2}, TC = {:.2}, seg_loss = {}'.format(val_ET, val_WT, val_TC, seg_loss))

    print('testing ...')
    test_ET, test_WT, test_TC, seg_loss = validate(args, input_size, model, testloader, args.num_classes, loss_D, loss_BCE)
    print('Testing ET = {:.2}, WT = {:.2}, TC = {:.2}, seg_loss = {}'.format(test_ET, test_WT, test_TC, seg_loss))



if __name__ == '__main__':
    main()
