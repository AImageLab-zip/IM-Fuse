import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
import numpy as np
import os
import time
import SimpleITK as sitk
import datetime
import argparse
import random
from PIL import Image
from tqdm import tqdm
import wandb
from pathlib import Path

from loader.Dataloader import get_loaders
from utils.util import check_dirs, print_net, re_crop, MulticlassDiceLoss
from loss import get_dice, U_Hemis_loss, RMBTS_loss, LMCR_loss
from net.Network_InOut import U_Hemis3D, TF_U_Hemis3D, AF_U_Hemis3D,InOut_U_Hemis3D, RsInOut_U_Hemis3D
from net.Network_RMBTS import RMBTS, TF_RMBTS
from net.Network_LMCR import LMCR, TF_LMCR
from process.utils import parse_image_name,missing_list

class Solver:
    def __init__(self, data_files, opt,base_dir):
        self.opt = opt
        print(self.opt)
        self.best_epoch = 0
        self.best_dice = 0
        self.best_epoch_extra = 0
        self.best_dice_extra = 0

        self.further_train = self.opt.resume
        self.further_epoch = self.opt.further_epoch
        self.trans = self.opt.trans
        self.model_name = self.opt.model_name
        self.TF_methods = self.opt.TF_methods
        self.phase = self.opt.phase
        self.out_channels = self.opt.out_channels
        self.in_channels = self.opt.in_channels
        self.levels = self.opt.levels
        self.feature_maps = self.opt.feature_maps
        self.selected_modal = self.opt.selected_modal
        self.batch_size = self.opt.batch_size
        self.num_workers = self.opt.num_workers
        loaders = get_loaders(base_dir, data_files, self.selected_modal, self.batch_size, self.num_workers)
        print('Get Loader')
        self.loaders = {x: loaders[x] for x in ('train', 'val', 'test')}

        self.c_dim = len(self.selected_modal)

        self.max_epoch = self.opt.num_epochs
        self.decay_epoch = self.opt.decay_epoch
        self.lr = self.opt.lr
        self.min_lr = self.opt.min_lr
        self.beta1 = self.opt.beta1
        self.beta2 = self.opt.beta2
        self.ignore_index = self.opt.ignore_index
        self.seg_loss_type = self.opt.seg_loss_type
        self.n_critic = self.opt.n_critic
        self.miss_list = missing_list()
        self.mdice = MulticlassDiceLoss()

        self.test_epoch = self.opt.test_epoch
        # #self.train_epoch = self.opt.train_epoch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.checkpoint_dir = self.opt.checkpoint_path
        self.sample_dir = os.path.join(self.checkpoint_dir, 'sample_dir')
        self.model_save_dir = os.path.join(self.checkpoint_dir, 'model_save_dir')
        self.result_dir = os.path.join(self.checkpoint_dir, 'result_dir')

        check_dirs([self.model_save_dir, self.result_dir, self.sample_dir, self.checkpoint_dir])
        print('Checked DIR')

        self.log_step = self.opt.log_step
        self.val_epoch = self.opt.val_epoch
        self.lr_update_epoch = self.opt.lr_update_epoch
        self.G = None
        self.build_model()
        print('......')

    def build_model(self):
        print('Building...')
        print('model_name==',self.model_name)
        self.G = eval(self.model_name)(in_channels=self.in_channels, out_channels=self.out_channels,
                           levels=self.levels, feature_maps=self.feature_maps, method=self.TF_methods, phase=self.phase)
        print('111')
        if self.phase == 'train':
            print('222')
            print_net(self.G, self.model_name)
        self.G.to(self.device)

        self.g_optimizer = optim.AdamW(self.G.parameters(),self.lr,[self.beta1,self.beta2],weight_decay=0.0001)


    def save_model(self, save_iters):
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(save_iters))
        last_path = os.path.join(self.model_save_dir, 'last-G.ckpt')
        checkpoint = {
        'model': self.G.state_dict(),
        'optimizer': self.g_optimizer.state_dict(),
        'epoch': save_iters
        }
        torch.save(checkpoint,G_path)
        torch.save(checkpoint,last_path)
        print('Saved model checkpoints into {}...'.format(self.model_save_dir))

    def restore_model(self):
        G_path = os.path.join(self.model_save_dir, 'last-G.ckpt')
        checkpoint = torch.load(G_path, map_location=self.device)
        self.G.load_state_dict(checkpoint['model'])
        self.g_optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch']

    def update_lr(self, lr):
        for param_group in self.g_optimizer.param_groups:
           param_group['lr'] = lr

    def reset_grad(self):
        self.g_optimizer.zero_grad()

    def clip_grad(self):
        nn.utils.clip_grad_norm_(self.G.parameters(), 20, 2)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def min_max(self, x):
        mi, ma = x.min(), x.max()
        x = (x - mi) / (ma - mi)
        return x

    @staticmethod
    def classification_loss(logit, target):
        return F.cross_entropy(logit, target)

    def check_m_d(self,m_d):
        n = m_d[0]
        for i in range(len(m_d)):
            if n != m_d[i]:
                print('error for different m_d in one batch!')

    def replace_modality(self, inputs, m_d):
        sum_modal = None
        num = 0
        for i in range(len(inputs)):
            if m_d in self.miss_list[i]:
                if sum_modal is None:
                    sum_modal = inputs[i]
                else:
                    sum_modal += inputs[i]
                num+=1
        aver = sum_modal/num
        for i in range(len(inputs)):
            if m_d not in self.miss_list[i]:
                inputs[i] = aver.clone()
        return inputs

    def train(self):
        loaders = {}
        loaders['train'] = self.loaders['train']
        start_epoch = 0
        cur_step = -1

        print('\nStart training...')
        if self.further_train:
            start_epoch = self.restore_model() +1
        self.lr = self.g_optimizer.param_groups[0]['lr']

        print('\nStart training...')
        start_time = time.time()

        for epoch in range(start_epoch, self.max_epoch):
            self.G.train()
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            for p in loaders.keys():
                for i, batch_data in tqdm(enumerate(loaders[p]),total=len(loaders[p])):
                    cur_step += 4
                    loss = {}

                    volume_label = batch_data[4].unsqueeze(dim=1).type(torch.float32).to(self.device)
                    pid = batch_data[5]
                    m_d = batch_data[6].to(self.device)
                    inputs = []
                    if (i + 1) % self.n_critic == 0:
                        for k in range(len(self.selected_modal)):
                            inputs.append(batch_data[k].unsqueeze(dim=1).type(torch.float32).to(self.device))

                        if self.model_name == ['LMCR']:
                            inputs = self.replace_modality(inputs, m_d[0])

                        seg, g_loss = None, None
                        if 'Hemis3D' in self.model_name:
                            seg = self.G(inputs, m_d[0])
                            g_loss = U_Hemis_loss(seg, volume_label, self.mdice)
                        elif self.model_name in ['RMBTS', 'TF_RMBTS']:
                            re_dic = self.G(inputs, m_d[0])
                            seg = re_dic['seg']

                            g_loss, dice_loss, ce_loss, rec_loss, KL_loss = RMBTS_loss(re_dic, volume_label, inputs,
                                                                                           self.mdice, m_d[0],
                                                                                           self.miss_list, self.device)

                            loss['dice'] = dice_loss.item()
                            loss['ce'] = ce_loss.item()
                            loss['rec'] = rec_loss.item()
                            loss['KL'] = KL_loss.item()

                        elif self.model_name in ['LMCR', 'TF_LMCR']:
                            re_dic = self.G(inputs, m_d[0])
                            seg = re_dic['seg']
                            g_loss, dice_loss, rec_loss = LMCR_loss(re_dic, volume_label, inputs,
                                                                                          self.mdice, m_d[0],
                                                                                          self.miss_list, self.device)
                            loss['dice'] = dice_loss.item()
                            loss['rec'] = rec_loss.item()
                        else:
                            print('error methods!!')


                        dice_wt, dice_tc, dice_et = get_dice(seg, volume_label)

                        self.reset_grad()
                        g_loss.backward()
                        self.clip_grad()
                        self.g_optimizer.step()
                        
                        loss['G/s'] = g_loss.item()
                        if torch.isnan(g_loss).any():
                            print('seg nan:', torch.isnan(seg).any())
                            print('label nan:', torch.isnan(volume_label).any())
                            return
                        loss['dc_wt'] = dice_wt
                        loss['dc_tc'] = dice_tc
                        loss['dc_et'] = dice_et
                        loss['pid'] = pid[0]
                        loss['m_d'] = int(m_d[0])

                    # =================================================================================== #
                    #                                 4. Miscellaneous                                    #
                    # =================================================================================== #
                    if (cur_step + 1) % self.log_step and self.opt.wandb_project_name is not None:
                            wandb.log({
                            "epoch": epoch + 1,
                            "iter": cur_step,
                            "loss/G": loss.get("G/s", None),
                            "dice/wt": loss.get("dc_wt", None),
                            "dice/tc": loss.get("dc_tc", None),
                            "dice/et": loss.get("dc_et", None),
                            "pid": loss.get("pid", None),
                            "missing_modality": loss.get("m_d", None),
                            "lr": self.g_optimizer.param_groups[0]["lr"]
                        })

            if (epoch + 1) % self.val_epoch == 0:
                print('\n')
                d1,d2,d3,dps = self.val(epoch + 1)
                print('Current dps of validation WT: {:.4f} TC: {:.4f} ET: {:.4f} Aver: {:.4f}'.format(d1,d2,d3,dps))

            if (epoch + 1) % self.lr_update_epoch == 0 and (epoch + 1) > (self.max_epoch - self.decay_epoch):
                dlr = self.lr - self.min_lr
                self.lr -= dlr / (self.decay_epoch / self.lr_update_epoch)
                self.update_lr(self.lr)
                print('Decayed learning rates, lr: {}.'.format(self.lr))




    def val(self, epoch):
        save_dir = os.path.join(self.result_dir, str(epoch))
        print('Start validation at iter {}...'.format(epoch))
        self.G.eval()
        loaders = {}
        loaders['val'] = self.loaders['val']
        d1, d2, d3 = 0, 0, 0
        n = 0
        with torch.no_grad():
            for p in loaders.keys():
                vis_index = []
                for k in range(10):
                    vis_index.append(random.randint(0, len(loaders[p]) - 1))
                for i, batch_data in enumerate(loaders[p]):

                    volume_label = batch_data[4].unsqueeze(dim=1).type(torch.cuda.FloatTensor).to(self.device).detach()

                    m_d = batch_data[6].to(self.device).detach()
                    self.check_m_d(m_d)

                    inputs = []
                    for k in range(len(self.selected_modal)):
                        inputs.append(batch_data[k].unsqueeze(dim=1).type(torch.cuda.FloatTensor).to(self.device).detach())
                    if self.model_name in ['LMCR']:
                        inputs = self.replace_modality(inputs, m_d[0])
                    if 'Hemis3D' in self.model_name:
                        seg = self.G(inputs, m_d[0])
                    else:
                        re_dic = self.G(inputs, m_d[0])
                        seg = re_dic['seg']

                    dice_wt, dice_tc, dice_et = get_dice(seg, volume_label)
                    d1, d2, d3 = d1+dice_wt, d2+dice_tc, d3+dice_et
                    n+=1

        dps = (d1+d2+d3)/(3*n)

        if self.best_dice<dps:
            self.best_epoch = epoch
            self.best_dice = dps
        self.save_model(epoch)
        print('Current best dps : {:.4f} of epoch : {}'.format(self.best_dice,self.best_epoch))
        return d1/n, d2/n, d3/n, dps

    def infer(self, epoch, method='forward'):
        loaders = {}
        loaders['test'] = self.loaders['test']
        save_dir = os.path.join(self.result_dir, str(epoch))
        check_dirs(save_dir)
        self.restore_model(epoch)
        self.G.eval()
        num_max = 69 * 15
        # check_dirs('/home/psdz/workplace/Aiyan/TFusion/heatmap')
        bbb = 0
        with torch.no_grad():
            for p in loaders.keys():
                for i, batch_data in enumerate(loaders[p]):

                    pid = batch_data[5]
                    m_d = batch_data[6].to(self.device)
                    crop_size = batch_data[7]
                    print('{}/{}\t p_id:{}\t m_id:{}'.format(i, num_max, pid[0], int(m_d[0])))
                    inputs = []
                    for k in range(len(self.selected_modal)):
                        inputs.append(
                            batch_data[k].unsqueeze(dim=1).type(torch.cuda.FloatTensor).to(self.device).detach())
                    if self.model_name in ['LMCR']:
                        inputs = self.replace_modality(inputs, m_d[0])
                    if self.model_name in ['TF_U_Hemis3D', 'U_Hemis3D', 'RsInOut_U_Hemis3D']:
                    #if 'RSInOut_U_Hemis3D' in self.model_name:
                        seg = self.G(inputs, m_d[0])
                    else:
                        re_dic  = self.G(inputs, m_d[0])
                        seg = re_dic['seg']

                    seg = F.softmax(seg, dim=1)
                    _, pr = torch.max(seg, dim=1, keepdim=True)
                    #pr[pr == 3] = 4
                    pre = np.array(re_crop(pr.squeeze(dim=0).squeeze(dim=0), crop_size).cpu()).astype(np.float)
                    out = sitk.GetImageFromArray(pre)
                    sitk.WriteImage(out, save_dir + '/{}_{}.nii.gz'.format(pid[0], int(m_d[0])))


        return

if __name__ == '__main__':
    cudnn.benchmark = True

    args = argparse.ArgumentParser()
    args.add_argument('--model_name', type=str, default='RsInOut_U_Hemis3D', help='Model name')  # Define model_name
    args.add_argument('--phase', type=str, default='train', help='Phase (train or test)')  # Define phase argument
    args.add_argument('--trans', type=str, default='trans')
    args.add_argument('--train_list', type=str, default='./process/partition/0-train.txt')
    args.add_argument('--val_list', type=str, default='./process/partition/0-val.txt')
    args.add_argument('--test_list', type=str, default='./process/partition/0-test.txt')
    
    args.add_argument('--selected_modal', nargs='+', default=['t1c', 't1n', 't2w', 't2f'])
    args.add_argument('--batch_size', type=int, default=1)
    args.add_argument('--out_channels', type=int, default=4)
    args.add_argument('--in_channels', type=int, default=1)
    args.add_argument('--feature_maps', type=int, default=8)
    args.add_argument('--levels', type=int, default=4)
    args.add_argument('--norm_type', type=str, default='instance')
    args.add_argument('--use_dropout', type=bool, default=True)

    args.add_argument('--decay_epoch', type=int, default=0)
    args.add_argument('--lr', type=float, default=3e-4)
    args.add_argument('--min_lr', type=float, default=1e-6)
    args.add_argument('--beta1', type=float, default=0.9)
    args.add_argument('--beta2', type=float, default=0.999)
    args.add_argument('--ignore_index', type=int, default=None)
    args.add_argument('--seg_loss_type', type=str, default='cross-entropy')
    args.add_argument('--seed', type=int, default=1234)
    args.add_argument('--use_weight', type=bool, default=True)
    args.add_argument('--n_critic', type=int, default=1)

    args.add_argument('--method', type=str, default='forward')

    args.add_argument('--device', type=bool, default=True)
    args.add_argument('--gpu_id', type=str, default='0')

    args.add_argument('--train_epoch', nargs='+', default=['50','100','150','200'])
    args.add_argument('--test_epoch', nargs='+', default=['50','100','150','200'])
    args.add_argument('--further_train', type=bool, default=False)
    args.add_argument('--further_epoch', type=int, default=0)
    
    args.add_argument('--TF_methods', type=str, default='TF')


    args.add_argument('--log_step', type=int, default=300)
    args.add_argument('--val_epoch', type=int, default=25)
    args.add_argument('--lr_update_epoch', type=int, default=1000)

    args.add_argument('--datapath', type=str, required=True)
    args.add_argument('--checkpoint-path', type=str, required=True)
    args.add_argument('--num-epochs', type=int, required=True)
    args.add_argument('--wandb-project-name',type=str,default=None)
    args.add_argument('--num-workers', type=int, default=0)
    args.add_argument('--resume',action='store_true')
    args = args.parse_args()
    print('-----Config-----')
    for k, v in sorted(vars(args).items()):
        print('%s:\t%s' % (str(k), str(v)))
    print('-------End------\n')


    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.wandb_project_name is not None:
        wandb.init(
            project='InOutFusion',
            name = 'training'
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    datalist_dir = Path(__file__).parent / 'datalist' 
    data_files = dict(train = datalist_dir/'train.txt', val = datalist_dir/'val.txt', test = datalist_dir/'test.txt')
    args.checkpoint_path = args.checkpoint_path + args.model_name
    assert args.batch_size == 1, 'This script only supports batch size 1'
    solver = Solver(data_files, args,args.datapath)
    if args.phase == 'train':
        solver.train()
    elif args.phase == 'test':
        print('calculating...')
        for test_iter in args.test_epoch:
            test_iter = int(test_iter)
            solver.infer(test_iter, args.method)

    print('Done!')
    
    
    

