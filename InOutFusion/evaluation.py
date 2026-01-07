import SimpleITK as sitk
import os
import numpy as np
import copy
import math
import argparse
import logging

def score(result_dir, selected_modal, selected_epoch, gt_dir, seg_type, print_func=print):
    # selected_epoch = selected_epoch[1]
    pred_dir = result_dir + '/' + selected_epoch
    print(result_dir, selected_epoch, seg_type)


    metrics = ['Dice']
    info = {x: {k: 0 for k in metrics}
            for x in selected_modal}
    n_sample = {x: 0 for x in selected_modal}
    modal_pid = {x: [] for x in selected_modal}

    a = [[]]
    aver = {}
    for modal in selected_modal:
        aver[modal] = copy.deepcopy(a)
    mid = {'Dice': 0}
    print_func('calculating...')
    ni = 0
    for volumn_name in os.listdir(pred_dir):
        p_id, modal = volumn_name.split('.')[0].split('_')
        modal = int(modal)
        if modal not in selected_modal:
            continue
        ni += 1
        predv_dir = os.path.join(pred_dir, volumn_name)
        pred_volume = sitk.GetArrayFromImage(sitk.ReadImage(predv_dir)).astype(np.int)

        gtv_dir = gt_dir + '/BraTS20_Training_{}/BraTS20_Training_{}_seg.nii.gz'.format(p_id, p_id)
        gt_volumn = sitk.GetArrayFromImage(sitk.ReadImage(gtv_dir)).astype(np.int)

        new_pred_volumn = pred_volume.copy()
        new_gt_volumn = gt_volumn.copy()
        if seg_type == 'ET':
            new_gt_volumn[new_gt_volumn == 2] = 0
            new_gt_volumn[new_gt_volumn == 1] = 0
            new_pred_volumn[new_pred_volumn == 2] = 0
            new_pred_volumn[new_pred_volumn == 1] = 0
        elif seg_type == 'TC':
            new_gt_volumn[new_gt_volumn == 2] = 0
            new_pred_volumn[new_pred_volumn == 2] = 0
        elif seg_type != 'WT':
            print('******************************************error type!!')
        new_gt_volumn[new_gt_volumn != 0] = 1
        new_pred_volumn[new_pred_volumn != 0] = 1

        if new_gt_volumn.sum() == 0:
            continue
        if new_pred_volumn.sum() == 0:

            dice = 0.0
            n_sample[modal] += 1
            info[modal]['Dice'] += dice
            aver[modal][mid['Dice']].append(dice)
            continue

        pred = sitk.GetImageFromArray(new_pred_volumn, isVector=False)
        gt = sitk.GetImageFromArray(new_gt_volumn, isVector=False)

        over_filter = sitk.LabelOverlapMeasuresImageFilter()
        over_filter.Execute(gt, pred)
        dice = over_filter.GetDiceCoefficient()

        n_sample[modal] += 1
        modal_pid[modal].append(p_id)

        info[modal]['Dice'] += dice
        aver[modal][mid['Dice']].append(dice)


    allinfo = {}
    selected_metric = ['Dice']
    for modal in selected_modal:
        s = info[modal]
        t = aver[modal]
        dic = {}
        for k, v in s.items():
            if k not in selected_metric:
                continue
            av = v / n_sample[modal]
            sum = 0
            for p in t[mid[k]]:
                sum += math.pow(p - av, 2)
            if n_sample[modal] != len(t[mid[k]]):
                assert 1>2, 'error and break'
            variance = sum / n_sample[modal]
            sd = math.sqrt(variance)
            if k in selected_metric:
                dic[k] = [av, sd]
        allinfo[modal] = dic
    sum_sample = 0
    print_str = '\t'
    for modal in selected_modal:
        sum_sample += n_sample[modal]
        print_str += '{}({}):\t'.format(modal, n_sample[modal])
    print_str += 'overall\n'
    for metric in selected_metric:
        aver = 0
        variance = 0
        print_str += '{}:\t'.format(metric)
        for modal in selected_modal:
            a = allinfo[modal][metric][0]
            aver += a * n_sample[modal]
            print_str += '{:.2f}\t'.format((a * 100))
        aver = aver / sum_sample

        print_str += '{:.2f}\n'.format((aver * 100))
    print_func(result_dir+selected_epoch)
    print_func(print_str)
    print(print_str)
    print_func('*' * 120 + '  {}'.format(seg_type))

if __name__ == '__main__':
    args = argparse.ArgumentParser('Compute the static between predictions and ground truth.')
    args.add_argument('--selected_epoch',nargs='+',default=['50','100','150','200'])
    # args.add_argument('--model_name', type=str, default='TF_U_Hemis3D')
    #args.add_argument('--model_name', type=str, default='ffTF_U_Hemis3D')
    # args.add_argument('--result_dir', type=str, default='./checkpoint/result_dir')
    args.add_argument('--model_name', type=str, default='InOut_U_Hemis3D')
    args.add_argument('--gt_dir', type=str, default='./dataset/MICCAI_BraTS2020_TrainingData') 
    opt = args.parse_args()
    # opt.result_dir = '/Code/SFusion-main/checkpoint/TF_RMBTS/result_dir'
    opt.result_dir = './checkpoint/{}/result_dir'.format(opt.model_name)
    print(opt.result_dir)
    selected_modal = [1,2,4,8,3,5,9,6,10,12,7,11,13,14,15]



    for sep in opt.selected_epoch:
        logger = logging.getLogger('my_log')
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler('./checkpoint/{}/result_dir/{}.log'.format(opt.model_name, sep))
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # print('./checkpoint/{}/result_dir/{}.log'.format(opt.model_name, sep))

        score(opt.result_dir, selected_modal, sep, opt.gt_dir, 'WT', print_func=logger.info)
        score(opt.result_dir, selected_modal, sep, opt.gt_dir, 'TC', print_func=logger.info)
        score(opt.result_dir, selected_modal, sep, opt.gt_dir, 'ET', print_func=logger.info)




#python evaluation.py --model_name InOut_U_Hemis3D