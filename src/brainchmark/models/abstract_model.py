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
        pass
