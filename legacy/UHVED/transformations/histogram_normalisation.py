# -*- coding: utf-8 -*-
"""
This class computes histogram based normalisation. A `training`
process is first used to find an averaged histogram mapping
from all training volumes.  This layer maintains the mapping array,
and the layer_op maps the intensity of new volumes to a normalised version.
The histogram is computed from foreground if a definition is provided for
foreground (by `binary_masking_func` or a `mask` matrix)
"""
from __future__ import absolute_import, print_function, division
import os
import numpy as np
from .histogram_standardisation import create_mapping_from_multimod_arrayfiles, transform_by_mapping, read_mapping_file, write_all_mod_mapping
from .mean_variance_normalisation import BinaryMaskingLayer
import torch.nn as nn


class HistogramNormalisationLayer(nn.Module):
    def __init__(self,
                 image_name,
                 modalities,
                 model_filename=None,
                 binary_masking_func=None,
                 norm_type='percentile',
                 cutoff=(0.05, 0.95),
                 name='hist_norm'):
        """

        :param image_name: Name of the image field to process.
        :param modalities: List of image modalities to normalize.
        :param model_filename: Path to the file storing normalization mappings.
        :param binary_masking_func: Optional function to generate binary masks 
                                    for selecting foreground regions.
        :param norm_type: Type of normalization ('percentile', etc.).
        :param cutoff: Percentile cutoff for normalization (low, high).
        :param name: Name of the layer.
        """

        super(HistogramNormalisationLayer, self).__init__()

        # Set default model file path if not provided
        if model_filename is None:
            model_filename = os.path.join('.', 'histogram_ref_file.txt')
        self.model_file = os.path.abspath(model_filename)
        assert not os.path.isdir(self.model_file), \
            "model_filename is a directory, " \
            "please change histogram_ref_file to a filename."
        
        # Set binary masking function if provided, else default to None
        if binary_masking_func:
            assert isinstance(binary_masking_func, BinaryMaskingLayer)
            self.binary_masking_func = binary_masking_func
        else:
            self.binary_masking_func = None

        # Set normalization type and cutoff values
        self.norm_type = norm_type
        self.cutoff = cutoff

        # Cache the normalization mappings from the model file
        # mapping is a complete cache of the model file, the total number of
        # modalities are listed in self.modalities tuple
        self.image_name = image_name
        self.modalities = modalities
        self.mapping = read_mapping_file(self.model_file)

    def forward(self, image, mask=None):
        """
        Apply histogram normalization to the input image.

        :param image: Input image (either dictionary or array).
        :param mask: Optional mask to specify regions of interest.
        :return: Normalized image and updated mask.
        """

        assert self.is_ready(), \
            "histogram normalisation layer needs to be trained first."
        
        # Convert image to NumPy array if it is a dictionary
        if isinstance(image, dict):
            image_5d = np.asarray(image[self.image_name], dtype=np.float32)
        else:
            image_5d = np.asarray(image, dtype=np.float32)

        # Retrieve or generate mask
        if isinstance(mask, dict):
            image_mask = mask.get(self.image_name, None)
        elif mask is not None:
            image_mask = mask
        elif self.binary_masking_func is not None:
            image_mask = self.binary_masking_func(image_5d)
        else:
            # no access to mask, default to all image
            image_mask = np.ones_like(image_5d, dtype=np.bool)

        # Apply normalization to the image
        normalised = self._normalise_5d(image_5d, image_mask)

        # Update image and mask if the input was a dictionary
        if isinstance(image, dict):
            image[self.image_name] = normalised
            if isinstance(mask, dict):
                mask[self.image_name] = image_mask
            else:
                mask = {self.image_name: image_mask}
            return image, mask
        else:
            return normalised, image_mask

    def __check_modalities_to_train(self):
        modalities_to_train = [mod for mod in self.modalities
                               if mod not in self.mapping]
        return set(modalities_to_train)

    def is_ready(self):
        mod_to_train = self.__check_modalities_to_train()
        return False if mod_to_train else True

    def train(self, image_list):
        # check modalities to train, using the first subject in subject list
        # to find input modality list
        if self.is_ready():
            print(
                "normalisation histogram reference models ready"
                " for {}:{}".format(self.image_name, self.modalities))
            return

        # Determine which modalities need training
        mod_to_train = self.__check_modalities_to_train()
        print(
            "training normalisation histogram references "
            "for {}:{}, using {} subjects".format(
                self.image_name, mod_to_train, len(image_list)))

        # Create normalization mappings from the training images
        trained_mapping = create_mapping_from_multimod_arrayfiles(
            image_list,
            self.image_name,
            self.modalities,
            mod_to_train,
            self.cutoff,
            self.binary_masking_func)
        
        # Update the mappings and save them to the model file
        # merging trained_mapping dict and self.mapping dict
        self.mapping.update(trained_mapping)
        all_maps = read_mapping_file(self.model_file)
        all_maps.update(self.mapping)
        write_all_mod_mapping(self.model_file, all_maps)

    def _normalise_5d(self, data_array, mask_array):
        """
        Normalize a 5D data array using the precomputed mappings.

        :param data_array: Input image data.
        :param mask_array: Mask for foreground regions.
        :return: Normalized data array.
        """

        assert self.modalities
        assert data_array.ndim == 5
        assert data_array.shape[4] <= len(self.modalities)

        if not self.mapping:
            print(
                "calling normaliser with empty mapping,"
                "probably {} is not loaded".format(self.model_file))
            raise RuntimeError
        mask_array = np.asarray(mask_array, dtype=np.bool)
        for mod_id, mod_name in enumerate(self.modalities):
            if not np.any(data_array[..., mod_id]):
                continue  # missing modality
            # Normalize the modality using the precomputed mapping
            data_array[..., mod_id] = self.__normalise(
                data_array[..., mod_id],
                mask_array[..., mod_id],
                self.mapping[mod_name])
        return data_array

    def __normalise(self, img_data, mask, mapping):
        """
        Normalize a single modality using the mapping and cutoff values.

        :param img_data: Input image data for a specific modality.
        :param mask: Foreground mask.
        :param mapping: Precomputed normalization mapping.
        :return: Normalized image data.
        """
        return transform_by_mapping(
            img_data, mask, mapping, self.cutoff, self.norm_type)
