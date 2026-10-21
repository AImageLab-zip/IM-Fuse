from __future__ import annotations

from collections.abc import Sequence

from mimose.models.tiny_mimosa import TinyMimosa

# Unlike TinyMimosa's whole-brain default (182, 218, 182, padded to fit),
# SimpleUnet trains/infers on fixed 128x128x128 patches -- the same tile
# size IM-Fuse-style transforms crop to (see
# mimose.training.transforms.imfuse.IMFuseTransformManager's default
# crop_size and cli_workflows.py's custom_trainer_kwargs.patch_size=128).
DEFAULT_INPUT_SHAPE = (128, 128, 128)


class SimpleUnet(TinyMimosa):
    """TinyMimosa's identical nnU-Net-style backbone (single joint encoder,
    mask-conditioned latent, sliding-window predict()), just sized for
    128x128x128-patch training instead of TinyMimosa's whole-volume
    pad-to-fit approach. Pair with transform_kind=imfuse (not
    transform_kind=tinymimosa) in custom_trainer_kwargs so the training
    transform actually random-crops to matching 128^3 patches -- see
    IMFuseTransformManager vs TinyMimosaTransformManager."""

    def __init__(
        self,
        *,
        input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
        **kwargs: object,
    ) -> None:
        super().__init__(input_shape=input_shape, **kwargs)


Model = SimpleUnet


__all__ = ["SimpleUnet", "Model"]
