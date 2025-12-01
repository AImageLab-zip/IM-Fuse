import torch
from testing import AverageMeter, test_softmax
from IMFuse import IMFuse
import os
import argparse
from pathlib import Path
import models
from data.datasets import BraTS_TrainCache, Cache
parser = argparse.ArgumentParser()

parser.add_argument('--dataname', default='BRATS2023', type=str)
parser.add_argument('--savepath', required=True, type=str)
parser.add_argument('--resume', required=True, type=str)
parser.add_argument('--datapath', required=True, type=str)
#parser.add_argument('--debug', action='store_true', default=False)
path = os.path.dirname(__file__)

test_file =Path(__file__).parent / 'datalist' / 'train15slits.csv'
if __name__ == '__main__':
    args = parser.parse_args()
    masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
    mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']

    datapath = args.datapath
    test_file = args.test_file
    save_path = Path(args.savepath)
    num_cls = 4
    dataname = args.dataname
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    
    test_cache = Cache(data_file_path=Path(args.datapath), splitfile=Path(__file__).parent / 'datalist' / 'test15slits.csv')
    test_set = BraTS_TrainCache(cache=test_cache, testing=True)
    model = models.DisenNet(
        inChans_list=[4],
        base_outChans=args.DisenNet_indim,
        num_class_list=[4],
    ).cuda()
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    output_path = save_path / 'results.txt'

    test_score = AverageMeter()
    with torch.no_grad():
        print('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks[index*5:(index+1)*5]):
            print('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                            test_loader,
                            model,
                            dataname = dataname,
                            feature_mask = mask,
                            compute_loss=False,
                            save_masks=True,
                            save_dir=save_path,
                            index = index)
            val_WT, val_TC, val_ET, val_ETpp = dice_score
            
            with open(output_path, 'a') as file:
                file.write('Performance missing scenario = {}, WT = {:.4f}, TC = {:.4f}, ET = {:.4f}, ETpp = {:.4f}\n'.format(mask, val_WT.item(), val_TC.item(), val_ET.item(), val_ETpp.item()))

            test_score.update(dice_score)
        print('Avg scores: {}'.format(test_score.avg))
        with open(output_path, 'a') as file:
                file.write('Avg scores: {}'.format(test_score.avg))
    test_cache.close()
