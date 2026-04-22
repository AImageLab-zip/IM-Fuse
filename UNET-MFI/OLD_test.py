from torch import autograd, optim
from torchvision.transforms import transforms as T
from Model import no_share_unet
from transforms import *
from Ddataset import BraTSDataset
from losses import *
from metrics import batch_dice_coef,dice_coef
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import gc
import argparse

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=3,
                        help='batch_size')

    #We saved the best model parameters on the validation set and the parameters at the later stage of trainging(at 900,950,1000,1050,1100,1150 epoch)
    #And the latter usually performs better on test set
    parser.add_argument('--bset_model_path', type=str, default="best_model/val_fold1_18_1050.pth")
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--patch_size', type=int, default=120,
                        help='patch_size for test')
    parser.add_argument('--overlap', type=int, default=40,
                        help='overlap for path')

    parser.add_argument('--use_TTA', type=bool, default=True,)
    parser.add_argument('--test_list', type=str, default='test1.txt')
    parser.add_argument('--data_path', type=str, default='/media/zzc/zzc12/pycharmproject/3D_miss_mod/2018/data/')
    parser.add_argument('--test_transforms', type=str, default='Compose([Pad((0, 0, 0, 5, 0)),NumpyType((np.float32, np.int64)),])')
    opt = parser.parse_args()
    return opt

def predict_dice(patch_size=80,overlap=40,use_TTA = True):

    opt = parse_option()
    model = no_share_unet(in_channel=1, out_channel=3, diff=True,deepSupvision=True).cuda()

    # dic = torch.load(opt.bset_model_path)
    # model.load_state_dict(dic['model.state'])

    model.load_state_dict(torch.load(opt.bset_model_path))

    Batch_Size = opt.batch_size
    test_list = opt.test_list
    test_data_dir =  opt.data_path
    test_transforms = opt.test_transforms

    H, W, T = 240, 240, 155
    masks = [[1, 1, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0], [0, 0, 1, 1], [0, 1, 0, 1],
             [0, 1, 1, 0], [1, 0, 0, 1], \
             [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    all_reslut = [[0.0]*4 for _ in range(15)]
    model.eval()
    gc.disable()

    with torch.no_grad():
        for index,mask in enumerate(masks):
            print("START TEST THE CODE：", mask,'-'*10)
            test_set = BraTSDataset(test_list, root=test_data_dir, mode='test', code=mask,
                                    transforms=test_transforms)
            test_loader = DataLoader(
                dataset=test_set,
                batch_size=Batch_Size,
                pin_memory=True, )

            WT_dice, TC_dice, ET_dice = [], [], []
            pos_ET_dice = []
            for i, (x1, x2, x3, x4, target, mask) in tqdm(enumerate(test_loader), total=len(test_loader)):  ##xi:b*1*240*240*160
                x1, x2, x3, x4 = x1.cuda(), x2.cuda(), x3.cuda(), x4.cuda()
                mask = mask.cuda()
                b,c,h,w,l = x1.shape
                cur_ret = torch.zeros((b,3,h,w,l)).cuda()
                cur_count = torch.zeros((b,3,h,w,l)).cuda()
                for row in range(0,240-patch_size+1,overlap):
                    for col in range(0,240-patch_size+1,overlap):
                        for height in range(0,160-patch_size+1,overlap):
                            cur_x1, cur_x2, cur_x3, cur_x4 = x1[:,:,row:row+patch_size,col:col+patch_size,height:height+patch_size],\
                                                             x2[:,:,row:row+patch_size,col:col+patch_size,height:height+patch_size],\
                                                             x3[:,:,row:row+patch_size,col:col+patch_size,height:height+patch_size],\
                                                             x4[:,:,row:row+patch_size,col:col+patch_size,height:height+patch_size]
                            if not use_TTA:
                                cur_output = model(cur_x1, cur_x2, cur_x3, cur_x4, mask)[-5:]
                                cur_ret[:, :, row:row + patch_size, col:col + patch_size, height:height + patch_size] += \
                                    (cur_output[0] + cur_output[1] + cur_output[2] + cur_output[3] + cur_output[4]) / 5
                            else:
                                cur_output = sum(model(cur_x1, cur_x2, cur_x3, cur_x4, mask)[-5:])/5
                                cur_output += (sum(model(cur_x1.flip(dims=(2,)), cur_x2.flip(dims=(2,)), cur_x3.flip(dims=(2,)), cur_x4.flip(dims=(2,)), mask)[-5:])/5).flip(dims=(2,))
                                cur_output += (sum(
                                    model(cur_x1.flip(dims=(3,)), cur_x2.flip(dims=(3,)), cur_x3.flip(dims=(3,)),
                                          cur_x4.flip(dims=(3,)), mask)[-5:]) / 5).flip(dims=(3,))
                                cur_output += (sum(
                                    model(cur_x1.flip(dims=(4,)), cur_x2.flip(dims=(4,)), cur_x3.flip(dims=(4,)),
                                          cur_x4.flip(dims=(4,)), mask)[-5:]) / 5).flip(dims=(4,))
                                cur_output += (sum(
                                    model(cur_x1.flip(dims=(2,3)), cur_x2.flip(dims=(2,3)), cur_x3.flip(dims=(2,3)),
                                          cur_x4.flip(dims=(2,3)), mask)[-5:]) / 5).flip(dims=(2,3))
                                cur_output += (sum(
                                    model(cur_x1.flip(dims=(2,4)), cur_x2.flip(dims=(2,4)), cur_x3.flip(dims=(2,4)),
                                          cur_x4.flip(dims=(2,4)), mask)[-5:]) / 5).flip(dims=(2,4))
                                cur_output += (sum(
                                    model(cur_x1.flip(dims=(3,4)), cur_x2.flip(dims=(3,4)), cur_x3.flip(dims=(3,4)),
                                          cur_x4.flip(dims=(3,4)), mask)[-5:]) / 5).flip(dims=(3,4))
                                cur_output += (sum(
                                    model(cur_x1.flip(dims=(2,3, 4)), cur_x2.flip(dims=(2,3, 4)), cur_x3.flip(dims=(2,3, 4)),
                                          cur_x4.flip(dims=(2,3, 4)), mask)[-5:]) / 5).flip(dims=(2,3, 4))

                                cur_output /= 8.0
                                cur_ret[:, :, row:row + patch_size, col:col + patch_size, height:height + patch_size] += cur_output
                            cur_count[:,:,row:row+patch_size,col:col+patch_size,height:height+patch_size] += 1
                cur_ret /= cur_count ##b*3*240*240*160
                cur_ret = torch.sigmoid(cur_ret)
                output = cur_ret[:, :, :H, :W, :T].cpu().numpy() ##b*3*240*240*155
                target = target[:, :, :H, :W, :T].numpy() ##b*3*240*240*155


                WT_out = output[:, 0, :, :, :] ##b*240*240*155
                WT_out[WT_out > 0.5] = 1
                WT_out[WT_out < 0.5] = 0

                TC_out = output[:, 1, :, :, :]
                TC_out[TC_out > 0.5] = 1
                TC_out[TC_out < 0.5] = 0

                ET_out = output[:, 2, :, :, :]  # b,240,240,155
                ET_out[ET_out > 0.5] = 1
                ET_out[ET_out < 0.5] = 0
                pos_ET_out = copy.deepcopy(ET_out)
                for i in range(b):
                   if pos_ET_out[i,...].sum() <=100:
                       pos_ET_out[i, ...] = np.zeros(pos_ET_out[i, ...].shape)


                WT_label = target[:, 0, :, :, :]
                TC_label = target[:, 1, :, :, :]
                ET_label = target[:, 2, :, :, :] # 240,240,155
                wt_dice = batch_dice_coef(WT_out, WT_label)
                et_dice = batch_dice_coef(ET_out, ET_label)
                pos_et_dice = batch_dice_coef(pos_ET_out, ET_label)
                tc_dice = batch_dice_coef(TC_out, TC_label)

                WT_dice.extend(wt_dice)
                ET_dice.extend(et_dice)
                pos_ET_dice.extend(pos_et_dice)
                TC_dice.extend(tc_dice)
                print('wt', wt_dice)
                print('et', et_dice)
                print('pos_et', pos_et_dice)
                print('tc', tc_dice)
            print('WT Dice: %.4f' % np.mean(WT_dice))
            print('TC Dice: %.4f' % np.mean(TC_dice))
            print('ET Dice: %.4f' % np.mean(ET_dice))
            print('POS_ET Dice: %.4f' % np.mean(pos_ET_dice))
            all_reslut[index][0] = np.mean(WT_dice)
            all_reslut[index][1] = np.mean(TC_dice)
            all_reslut[index][2] = np.mean(ET_dice)
            all_reslut[index][3] = np.mean(pos_ET_dice)
        print("all 15 code result:")
        for ret in all_reslut:
            print(ret)
        print('all 15 code mean result:')
        print(np.mean(all_reslut,axis=0))
        gc.enable()

if __name__ == '__main__':
    opt = parse_option()
    predict_dice(opt.patch_size,opt.overlap,use_TTA=opt.use_TTA)
