"""
Created on Wed Oct 14 17:38:59 2020

@author: reubendo
"""

"""
Adapted from https://github.com/taigw/brats17/blob/master/util/data_process.py

This script renames BRATS dataset to OUTPUT_path,
each subject's images will be cropped and renamed to
"TYPEindex_modality.nii.gz".

output dataset folder will be created if not exists, and content
in the created folder will be, for example:

OUTPUT_path:
   HGG100_Flair.nii.gz
   HGG100_Label.nii.gz
   HGG100_T1c.nii.gz
   HGG100_T1.nii.gz
   HGG100_T2.nii.gz
   ...

Each .nii.gz file in OUTPUT_path will be cropped with a tight bounding box
using function crop_zeros defined in this script.
"""
import os
import nibabel
import numpy as np
import argparse
import pandas as pd 
from transformations.mean_variance_normalisation import MeanVarNormalisationLayer, BinaryMaskingLayer
from transformations.histogram_normalisation import HistogramNormalisationLayer
from transformations.pad import PadLayer


mod_names17 = ['flair', 't1', 't1ce', 't2']
mod_names15 = ['Flair', 'T1', 'T1c', 'T2']
mod_names23 = ['t2f', 't1n', 't1c', 't2w']
SUPPORTED_INPUT = set(['image', 'label', 'weight', 'sampler', 'inferred', 'choices', 'output_mod'])
OUTPUT_AFFINE = np.array(
    [[-1, 0, 0, 0],
     [0, -1, 0, 239],
     [0, 0, 1, 0],
     [0, 0, 0, 1]])

def load_scans_BRATS(pat_folder, with_seg=False):
    nii_fnames = [f_name for f_name in os.listdir(pat_folder)
                  if f_name.endswith(('.nii', '.nii.gz')) 
                  and not f_name.startswith('._')]
    img_data = []
    for mod_n in mod_names23:
        file_n = [f_n for f_n in nii_fnames if (mod_n + '.') in f_n][0]
        mod_data = nibabel.load(os.path.join(pat_folder, file_n)).get_fdata()
        img_data.append(mod_data)
    img_data = np.stack(img_data, axis=-1) # (240, 240, 155, 4)
    if not with_seg:
        return img_data, None
    else:
        file_n = [f_n for f_n in nii_fnames if ('seg.') in f_n][0]
        seg_data = nibabel.load(os.path.join(pat_folder, file_n)).get_fdata() # (240, 240, 155)
        return img_data, seg_data

def save_image_BRATS(out_path, img_data):
    mod_data_nii = nibabel.Nifti1Image(img_data, affine=OUTPUT_AFFINE)
    nibabel.save(mod_data_nii, out_path)
    print('saved to {}'.format(out_path))

def save_scans_BRATS(out_path, pat_name, img_data, seg_data=None):
    save_mod_names = mod_names23
    save_seg_name = 'seg'
    assert img_data.shape[3] == 4
    for mod_i in range(len(save_mod_names)):
        save_name = '%s-%s.nii.gz' % (pat_name, save_mod_names[mod_i])
        save_path = os.path.join(out_path, save_name)
        mod_data_nii = nibabel.Nifti1Image(img_data[:, :, :, mod_i],
                                           OUTPUT_AFFINE)
        nibabel.save(mod_data_nii, save_path)
    print('saved to {}'.format(out_path))
    if seg_data is not None:
        save_name = '%s-%s.nii.gz' % (pat_name, save_seg_name)
        save_path = os.path.join(out_path, save_name)
        seg_data_nii = nibabel.Nifti1Image(seg_data, OUTPUT_AFFINE)
        nibabel.save(seg_data_nii, save_path)

def main(args, in_path, out_path, padding=False, norm_back_only=False, hist_norm=False, white_norm=False):
    data_path = '/work/grana_neuro/missing_modalities/UHVED/data/BRATS2023'
    train_path = '/work/grana_neuro/missing_modalities/UHVED/extensions/u_hved/datalist/train.txt'
    
    # initialise input preprocessing layers
    foreground_masking_layer = BinaryMaskingLayer(
        type_str=args.foreground_type,
        threshold=0.0) if norm_back_only else None
    
    mean_var_normaliser = MeanVarNormalisationLayer(
        image_name='image', binary_masking_func=foreground_masking_layer) if white_norm else None
    
    histogram_normaliser = HistogramNormalisationLayer(
        image_name='image',
        modalities=mod_names23,
        model_filename=args.histogram_ref_file,
        binary_masking_func=foreground_masking_layer,
        cutoff=args.cutoff,
        name='hist_norm_layer') if hist_norm else None
    
    pad_layer = PadLayer(
        image_name=SUPPORTED_INPUT,
        border=args.volume_padding_size) if padding else None
    
    with open(train_path, 'r') as f:
            datalist_train = [i.strip() for i in f.readlines()] #875 elements
    print(f"Length training set: {len(datalist_train)}")

    for pat_folder_name in os.listdir(os.path.join(in_path)): 
        case_id = pat_folder_name.split('/')[-1]
        img_data = []
        for i, mod_i in enumerate(mod_names23):
            img_name = '%s-%s.nii.gz' % (case_id, mod_i)
            mod_path = os.path.join(data_path, img_name)
            try:
                mod_data = nibabel.load(mod_path).get_fdata()
                img_data.append(mod_data)
            except OSError:
                print('skipping %s' % mod_path)
                continue
        img_data = np.stack(img_data, axis=-1)
        img_data = np.expand_dims(img_data, axis=3)
        print("subject: {}, shape: {}".format(pat_folder_name, img_data.shape))

        seg_name = '%s-%s.nii.gz' % (case_id, 'seg')
        seg_path = os.path.join(data_path, seg_name)
        seg_data = nibabel.load(os.path.join(seg_path)).get_fdata() #(240, 240, 155)
        
        ### Input preprocessing layers
        ### Padding (only to the training set)
        if padding and (case_id in datalist_train):
            img_data, mask = pad_layer(img_data)
            seg_data, mask = pad_layer(seg_data)
            print('shape image padding: {}'.format(img_data.shape))
            print('shape label padding: {}'.format(seg_data.shape))
        ### Histogram Normalization
        if hist_norm:
            img_data, image_mask = histogram_normaliser(img_data)
            print('shape hist norm: {}'.format(img_data.shape))
        ### zero mean and unit-variance normalization
        if white_norm:
            img_data, image_mask = mean_var_normaliser(img_data)
            print('mean and std after whitening norm: {}, {}'.format(img_data.mean(), img_data.std()))

        # save image
        save_scans_BRATS(out_path=out_path, pat_name=case_id, img_data=np.squeeze(img_data, axis=3), seg_data=seg_data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cropping scans')

    parser.add_argument('--input', '-i', help='Directory containing the BRATS dataset')
    parser.add_argument('--output', '-o', help='Directory containing the BRATS dataset')
    
    parser.add_argument('--foreground_type', type=str, default='mean_plus')
    parser.add_argument('--histogram_ref_file', type=str, default='/work/grana_neuro/missing_modalities/UHVED/extensions/u_hved/hist.txt')
    parser.add_argument('--cutoff', default=(0.001, 0.999))

    parser.add_argument('--volume_padding_size', default=(10, 10, 10))

    parser.add_argument('--whitening', type=bool, default=True)
    parser.add_argument('--normalisation', type=bool, default=True)
    parser.add_argument('--normalise_foreground_only', type=bool, default=True)
    parser.add_argument('--padding', type=bool, default=True)

    args = parser.parse_args()
    BRATS_path = args.input
    OUTPUT_path = args.output

    args = parser.parse_args()
    if not os.path.exists(BRATS_path):
        raise ValueError(
            'Dataset not found: {}'.format(BRATS_path))
    if not os.path.exists(OUTPUT_path):
        os.makedirs(OUTPUT_path)
    main(args, BRATS_path, OUTPUT_path, padding=args.padding, norm_back_only=args.normalise_foreground_only, hist_norm=args.normalisation, white_norm=args.whitening)
