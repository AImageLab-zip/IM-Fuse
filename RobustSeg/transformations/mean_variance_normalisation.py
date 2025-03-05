# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import numpy.ma as ma
import scipy.ndimage as ndimg
from scipy.ndimage.morphology import binary_fill_holes as fill_holes
import torch.nn as nn


"""
This class defines methods to generate a binary image from an input image.
The binary image can be used as an automatic foreground selector, so that later
processing layers can only operate on the `True` locations within the image.
"""
SUPPORTED_MASK_TYPES = set(['threshold_plus', 'threshold_minus',
                            'mean_plus'])

SUPPORTED_MULTIMOD_MASK_TYPES = set(['or', 'and', 'multi'])


class BinaryMaskingLayer(nn.Module):
    def __init__(self,
                 type_str='otsu_plus',
                 multimod_fusion='or',
                 threshold=0.0):
        super(BinaryMaskingLayer, self).__init__()
        self.type_str = type_str
        self.multimod_fusion = multimod_fusion
        #self.type_str = look_up_operations(
         #   type_str.lower(), SUPPORTED_MASK_TYPES)
        #self.multimod_fusion = look_up_operations(
         #   multimod_fusion.lower(), SUPPORTED_MULTIMOD_MASK_TYPES)
        self.threshold = threshold

    def __make_mask_3d(self, image):
        assert image.ndim == 3
        assert self.type_str in SUPPORTED_MASK_TYPES
        image_shape = image.shape
        image = image.reshape(-1)
        mask = np.zeros_like(image, dtype=np.bool)
        thr = self.threshold
        if self.type_str == 'threshold_plus':
            mask[image > thr] = True
        elif self.type_str == 'threshold_minus':
            mask[image < thr] = True
        elif self.type_str == 'mean_plus':
            thr = np.mean(image)
            mask[image > thr] = True
        mask = mask.reshape(image_shape)
        mask = ndimg.binary_dilation(mask, iterations=2)
        mask = fill_holes(mask)
        # foreground should not be empty
        assert np.any(mask == True), \
            "no foreground based on the specified combination parameters, " \
            "please change choose another `mask_type` or double-check all " \
            "input images"
        return mask

    def forward(self, image):
        if image.ndim == 3:
            return self.__make_mask_3d(image)

        if image.ndim == 5:
            mod_to_mask = [m for m in range(image.shape[4])
                           if np.any(image[..., :, m])]
            mask = np.zeros_like(image, dtype=bool)
            mod_mask = None
            for mod in mod_to_mask:
                for t in range(image.shape[3]):
                    mask[..., t, mod] = self.__make_mask_3d(image[..., t, mod])
                # combine masks across the modalities dim
                if self.multimod_fusion == 'or':
                    if mod_mask is None:
                        mod_mask = np.zeros(image.shape[:4], dtype=bool)
                    mod_mask = np.logical_or(mod_mask, mask[..., mod])
                elif self.multimod_fusion == 'and':
                    if mod_mask is None:
                        mod_mask = np.ones(image.shape[:4], dtype=bool)
                    mod_mask = np.logical_and(mod_mask, mask[..., mod])
            for mod in mod_to_mask:
                mask[..., mod] = mod_mask
            return mask
        else:
            raise ValueError("unknown input format")


class MeanVarNormalisationLayer(nn.Module):
    """
    This class defines image-level normalisation by subtracting
    foreground mean intensity value and dividing by standard deviation
    """

    def __init__(self, image_name, binary_masking_func=None):
        super(MeanVarNormalisationLayer, self).__init__()
        self.image_name = image_name
        self.binary_masking_func = None
        if binary_masking_func is not None:
            assert isinstance(binary_masking_func, BinaryMaskingLayer)
            self.binary_masking_func = binary_masking_func

    def forward(self, image, mask=None):
        if isinstance(image, dict):
            image_data = np.asarray(image[self.image_name], dtype=np.float32)
        else:
            image_data = np.asarray(image, dtype=np.float32)

        if isinstance(mask, dict):
            image_mask = mask.get(self.image_name, None)
        elif mask is not None:
            image_mask = mask
        elif self.binary_masking_func is not None:
            image_mask = self.binary_masking_func(image_data)
        else:
            # no access to mask, default to the entire image
            image_mask = np.ones_like(image_data, dtype=np.bool)

        if image_data.ndim == 3:
            image_data = whitening_transformation(image_data, image_mask)
        if image_data.ndim == 5:
            for m in range(image_data.shape[4]):
                for t in range(image_data.shape[3]):
                    image_data[..., t, m] = whitening_transformation(
                        image_data[..., t, m], image_mask[..., t, m])

        if isinstance(image, dict):
            image[self.image_name] = image_data
            if isinstance(mask, dict):
                mask[self.image_name] = image_mask
            else:
                mask = {self.image_name: image_mask}
            return image, mask
        else:
            return image_data, image_mask


def whitening_transformation(image, mask):
    # make sure image is a monomodal volume
    masked_img = ma.masked_array(image, np.logical_not(mask))
    image = (image - masked_img.mean()) / max(masked_img.std(), 1e-5)
    return image
