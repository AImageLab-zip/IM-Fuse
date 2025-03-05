import torch
from utils.predict import AverageMeter, test_softmax
from data.datasets_nii import Brats_loadall_test_nii
from utils.lr_scheduler import MultiEpochsDataLoader 
from UHVED import U_HVED


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
    datapath = '/work/grana_neuro/missing_modalities/BRATS2023_Training_npy'
    test_file = 'datalist/test15splits.csv'
    resume = '/work/grana_neuro/missing_modalities/UHVED/output_ROP4/best.pth'
    num_cls = 4
    dataname = 'BRATS2023'

    test_set = Brats_loadall_test_nii(transforms=test_transforms, root=datapath, test_file=test_file)
    test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = U_HVED(num_classes=num_cls)
    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])
    best_epoch = checkpoint['epoch'] + 1
    output_path = f"output_UHVED{best_epoch}.txt"

    test_score = AverageMeter()
    with torch.no_grad():
        print('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            print('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                            test_loader,
                            model,
                            dataname = dataname,
                            feature_mask = mask,
                            compute_loss=False)
            val_WT, val_TC, val_ET, val_ETpp = dice_score
            
            with open(output_path, 'a') as file:
                file.write('Performance missing scenario = {}, WT = {:.4f}, TC = {:.4f}, ET = {:.4f}, ETpp = {:.4f}\n'.format(mask, val_WT.item(), val_TC.item(), val_ET.item(), val_ETpp.item()))

            test_score.update(dice_score)
        print('Avg scores: {}'.format(test_score.avg))
        with open(output_path, 'a') as file:
                file.write('Avg scores: {}'.format(test_score.avg))
