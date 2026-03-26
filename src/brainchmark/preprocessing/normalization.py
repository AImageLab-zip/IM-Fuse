import numpy as np
from argparse import Namespace
import click

from brainchmark.preprocessing.config import NormConfig
def check_size(images:np.ndarray):
    if len(images.shape) != 4:
        raise click.ClickException(
            f"`images` must have 4 dimensions, got shape {images.shape}."
        )
def template(images:np.ndarray,config:NormConfig)-> np.ndarray:
    """Template function for intensity normalization on multi-modal 3D images.

    Args:
        images: Input image array with shape `(C, X, Y, Z)`, where `C` is the
            modality/channel dimension.
        config: Normalization configuration controlling the normalization behavior.

    Returns:
        The normalized image array with the same shape as `images`.
    """
    # implement your normalization logic here
    raise NotImplementedError()
    # return out_image
def subject_zscore(images:np.ndarray,config:NormConfig)-> np.ndarray:
    check_size(images)
    out_image = np.zeros_like(images)
    for i in range(images.shape[0]):
        foreground = images[i][images[i] > 0]
        mean = np.mean(foreground)
        std = np.std(foreground)
        out_image[i] = (images[i] - mean) / std
    return out_image



def min_max(images: np.ndarray, config: NormConfig) -> np.ndarray:
    check_size(images)

    target_min = config.min_max_range[0]
    target_max = config.min_max_range[1]

    out = images.copy()

    for i in range(out.shape[0]):
        single_modal = out[i]
        source_min = np.min(single_modal)
        source_max = np.max(single_modal)

        if source_max == source_min:
            raise click.ClickException(
                f"Cannot apply min-max normalization to modality {i}: "
                f"all values are equal to {source_min}."
            )

        out[i] = (
            (single_modal - source_min) / (source_max - source_min)
        ) * (target_max - target_min) + target_min

    return out

def dataset_zscore(images: np.ndarray, config: NormConfig) -> np.ndarray:
    check_size(images)
    out_image = np.zeros_like(images)
    for i in range(images.shape[0]):
        foreground = images[i][images[i] > 0]
        out_image[i] = (images[i] - config.mean[i]) / config.std[i]
    return out_image

def none(images: np.ndarray, config: NormConfig) -> np.ndarray:
    print("⚠️ Be careful: you are running preprocessing without any normalization.")
    return images
