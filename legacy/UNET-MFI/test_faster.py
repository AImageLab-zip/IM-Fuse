import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from test_utils import AverageMeter, softmax_output_dice_class4, set_seed
from Ddataset import BraTSDataset
from transforms import *
from Model import no_share_unet


def expand_mask_for_batch(mask, batch_size, device):
    """
    Ensure mask has shape [B, K] matching batch_size.
    Accepts numpy array, torch tensor of shape (K,) or (1, K) or (B, K).
    Uses repeat to produce independent rows (safe if model may modify).
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    mask = mask.to(device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)          # [1, K]
    if mask.size(0) != batch_size:
        mask = mask.repeat(batch_size, 1) # [B, K]
    return mask


def main():
    DEVICE = torch.device('cuda')
    set_seed(42)
    H, W, T = 240, 240, 155
    patch_size = 120
    overlap = 40
    use_TTA = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', required=True, type=Path)
    parser.add_argument('--savepath', required=True, type=Path)
    parser.add_argument('--resume', required=True, type=Path)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--micro-bs', default=8, type=int, help='number of patches per forward')
    parser.add_argument('--use-compile', action='store_true')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    masks = [
        [False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
        [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True],
        [True, False, False, True], [True, True, False, False], [True, True, True, False], [True, False, True, True],
        [True, True, False, True], [False, True, True, True], [True, True, True, True]
    ]

    ordered_names = ['t2f', 't1c', 't1n', 't2w']
    mask_names = ['_'.join([ordered_names[i] for i in range(4) if mask[i]]) for mask in masks]

    datapath = args.datapath
    test_file = Path(__file__).parent / 'datalist' / 'test.txt'

    model = no_share_unet(in_channel=1, out_channel=3, diff=True, deepSupvision=True).to(DEVICE)
    model.eval()
    checkpoint = torch.load(args.resume, weights_only=False)
    model.load_state_dict(checkpoint['model'])

    if args.use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    output_path = f"{args.savepath}"
    if os.path.exists(output_path):
        os.remove(output_path)
    total_score = AverageMeter()

    # Build one test loader with ALL modalities present (so we can apply arbitrary masks at runtime)
    test_set = BraTSDataset(
        test_file,
        root=datapath,
        mode='test',
        for_train=False,
        code=[1, 1, 1, 1],  # return full modalities; we will zero them per-mask in the loop
        transforms='Compose([Pad((0, 0, 0, 5, 0)),NumpyType((np.float32, np.int64)),])'
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    assert test_loader.batch_size == 1, 'keep batch size 1'

    # Precompute sliding-window coords
    coords = [(r, c, z)
              for r in range(0, 240 - patch_size + 1, overlap)
              for c in range(0, 240 - patch_size + 1, overlap)
              for z in range(0, 160 - patch_size + 1, overlap)]

    micro_bs = int(args.micro_bs)
    use_amp = True  # enable mixed precision for inference

    with torch.no_grad():
        for mi, mask_bool_list in tqdm(list(enumerate(masks)), desc='Evaluating all the masks'):
            mask_name = mask_names[mi]
            mask_arr = np.array(mask_bool_list).astype(np.int32)
            mask_tensor = torch.from_numpy(mask_arr).to(DEVICE)  # shape [4]

            mask_specific_score = AverageMeter()

            for _, (x1, x2, x3, x4, target, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
                # Move volumes to GPU once
                x1 = x1.to(DEVICE)  # shape [1,1,240,240,160]
                x2 = x2.to(DEVICE)
                x3 = x3.to(DEVICE)
                x4 = x4.to(DEVICE)
                # Apply the current mask (0/1 per modality) to the full volumes
                x1_masked = x1 * mask_tensor[0].view(1, 1, 1, 1, 1)
                x2_masked = x2 * mask_tensor[1].view(1, 1, 1, 1, 1)
                x3_masked = x3 * mask_tensor[2].view(1, 1, 1, 1, 1)
                x4_masked = x4 * mask_tensor[3].view(1, 1, 1, 1, 1)

                b, c, h, w, l = x1_masked.shape
                cur_ret = torch.zeros((b, 3, h, w, l), device=DEVICE, dtype=torch.float32)
                cur_count = torch.zeros((b, 3, h, w, l), device=DEVICE, dtype=torch.float32)

                # accumulate micro-batch patches
                patches1, patches2, patches3, patches4, locs = [], [], [], [], []

                # inference context with autocast
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    for (r, c, z) in coords:
                        patches1.append(x1_masked[:, :, r:r+patch_size, c:c+patch_size, z:z+patch_size])
                        patches2.append(x2_masked[:, :, r:r+patch_size, c:c+patch_size, z:z+patch_size])
                        patches3.append(x3_masked[:, :, r:r+patch_size, c:c+patch_size, z:z+patch_size])
                        patches4.append(x4_masked[:, :, r:r+patch_size, c:c+patch_size, z:z+patch_size])
                        locs.append((r, c, z))

                        # when micro-batch full (or last loop), forward
                        if len(patches1) >= micro_bs:
                            bx1 = torch.cat(patches1, dim=0)  # [micro_bs,1,ps,ps,ps]
                            bx2 = torch.cat(patches2, dim=0)
                            bx3 = torch.cat(patches3, dim=0)
                            bx4 = torch.cat(patches4, dim=0)

                            # expand mask to per-sample rows before forwarding
                            mask_expanded = expand_mask_for_batch(mask_tensor, bx1.size(0), DEVICE)  # [micro_bs, 4]

                            # model returns many items; last is cat_out in your architecture
                            outputs_batch = model(bx1, bx2, bx3, bx4, mask_expanded)[-1]  # shape [micro_bs,3,ps,ps,ps] (float16)
                            outputs_batch = outputs_batch.to(torch.float32)

                            for j, (rr, cc, zz) in enumerate(locs):
                                cur_ret[:, :, rr:rr+patch_size, cc:cc+patch_size, zz:zz+patch_size] += outputs_batch[j:j+1]
                                cur_count[:, :, rr:rr+patch_size, cc:cc+patch_size, zz:zz+patch_size] += 1.0

                            patches1, patches2, patches3, patches4, locs = [], [], [], [], []

                    # flush any remaining patches
                    if len(patches1) > 0:
                        bx1 = torch.cat(patches1, dim=0)
                        bx2 = torch.cat(patches2, dim=0)
                        bx3 = torch.cat(patches3, dim=0)
                        bx4 = torch.cat(patches4, dim=0)

                        mask_expanded = expand_mask_for_batch(mask_tensor, bx1.size(0), DEVICE)
                        outputs_batch = model(bx1, bx2, bx3, bx4, mask_expanded)[-1].to(torch.float32)
                        for j, (rr, cc, zz) in enumerate(locs):
                            cur_ret[:, :, rr:rr+patch_size, cc:cc+patch_size, zz:zz+patch_size] += outputs_batch[j:j+1]
                            cur_count[:, :, rr:rr+patch_size, cc:cc+patch_size, zz:zz+patch_size] += 1.0

                # normalize and sigmoid
                cur_ret /= cur_count
                cur_ret = torch.sigmoid(cur_ret)

                # permute/reshape exactly as in your original pipeline
                output = cur_ret[:, :, :H, :W, :T].squeeze(0).permute(0, 3, 1, 2)  # [C, T, H, W]
                target = target[:, :, :H, :W, :T].squeeze(0).permute(0, 3, 1, 2)

                _, brats_dice = softmax_output_dice_class4(output=output, target=target)
                mask_specific_score.update(brats_dice)

            mask_score_avg = mask_specific_score.avg
            total_score.update(mask_score_avg)
            mask_score_avg = mask_score_avg[0]
            with open(output_path, 'a') as file:
                file.write(f'Available modals = {mask_name:<21}--> WT = {mask_score_avg[0].item():.4f}, TC = {mask_score_avg[1].item():.4f}, ET = {mask_score_avg[2].item():.4f}, ETpp = {mask_score_avg[3].item():.4f}\n')

        avg_totalscore = total_score.avg[0]
        with open(output_path, 'a') as file:
            file.write(f'Avg scores {"":<29}--> WT = {avg_totalscore[0].item():.4f}, TC = {avg_totalscore[1].item():.4f}, ET = {avg_totalscore[2].item():.4f}, ETpp = {avg_totalscore[3].item():.4f}\n')


if __name__ == '__main__':
    main()