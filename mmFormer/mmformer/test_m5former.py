import torch
from predict import AverageMeter, test_softmax_limage
from data.datasets_nii import Brats_loadall_test_nii_loc
from utils.lr_scheduler import MultiEpochsDataLoader 
from m5former import Model
import torch.nn as nn


if __name__ == '__main__':
    masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
    mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
    
    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    datapath = '/work/grana_neuro/missing_modalities/mmFormer/BRATS2023_Training_npy'
    test_file = 'datalist/test15splits2.csv'
    resume = '/work/grana_neuro/missing_modalities/mmFormer/output/best.pth'
    num_cls = 4
    dataname = 'BRATS2023'

    test_set = Brats_loadall_test_nii_loc(transforms=test_transforms, root=datapath, test_file=test_file)
    test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = Model(num_cls=num_cls)
    model = torch.nn.DataParallel(model).cuda()
    limage = torch.randn((1, 4, 155, 240, 240))
    
    #limage reload
    ck = torch.load('/work/grana_neuro/missing_modalities/m3ae/runs/m3ae_pretrain/model_1model_best_599.pth.tar', map_location=torch.device('cpu'))
    with torch.no_grad():
        limage.copy_(ck['state_dict']['module.limage'])
    limage = limage.cuda()
    
    #model reload
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    best_epoch = checkpoint['epoch'] + 1
    output_path = f"output{best_epoch}_m5former_2.txt"

    test_score = AverageMeter()
    with torch.no_grad():
        print('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            print('{}'.format(mask_name[i]))
            dice_score = test_softmax_limage(
                            test_loader,
                            model,
                            dataname = dataname,
                            feature_mask = mask,
                            compute_loss=False, 
                            limage=limage)
            val_WT, val_TC, val_ET, val_ETpp = dice_score
            
            with open(output_path, 'a') as file:
                file.write('Performance missing scenario = {}, WT = {:.4f}, TC = {:.4f}, ET = {:.4f}, ETpp = {:.4f}\n'.format(mask, val_WT.item(), val_TC.item(), val_ET.item(), val_ETpp.item()))

            test_score.update(dice_score)
        print('Avg scores: {}'.format(test_score.avg))
        with open(output_path, 'a') as file:
                file.write('Avg scores: {}'.format(test_score.avg))
