from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractModel(nn.Module, ABC):
    @abstractmethod
    def predict(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run test-time inference for a batch of masked input volumes.

        This is the evaluation contract used by the BrainchMark testing
        pipeline. Unlike ``forward(...)``, which may return trainer-specific
        tuples for loss computation, ``predict(...)`` must return the final
        segmentation prediction tensor only.

        Expected inputs:
        - ``images``: tensor shaped ``[B, M, H, W, D]`` containing the input
          volumes for a batch, where ``M`` is the modality dimension.
        - ``mask``: boolean tensor describing which modalities are available
          for each sample. Implementations must apply the same modality logic
          used during training, including any internal modality reordering.

        Expected output:
        - a tensor shaped ``[B, C, H, W, D]`` containing segmentation logits
          or probabilities, where ``C`` is the number of output classes.

        Implementations may call ``forward(...)`` internally, run a dedicated
        sliding-window inference path, or disable training-only branches before
        producing the final prediction. The important point is that callers of
        ``predict(...)`` should not need to know any trainer-specific details.
        """
        pass
