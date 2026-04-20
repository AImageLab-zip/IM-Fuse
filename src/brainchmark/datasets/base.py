from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset

from brainchmark.datasets.config import MaskingMode


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        root: str | Path,
        split: list[dict[str, Any]],
        masking_mode: MaskingMode | None = None,
        mask_patterns: torch.Tensor | None = None,
        sample_transform: Callable[[torch.Tensor, torch.Tensor], tuple[Any, Any]] | None = None,
        image_transforms: Callable[[torch.Tensor], Any] | None = None,
        target_transforms: Callable[[torch.Tensor], Any] | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(f"Dataset directory not found: {self.root}")

        self.masking_mode = masking_mode
        self.mask_patterns = mask_patterns
        self.sample_transform = sample_transform
        self.image_transform = image_transforms
        self.target_transform = target_transforms
        self.samples = self._load_samples(split)

        if not self.samples:
            raise ValueError(f"No samples found in {self.root}")

    def _load_samples(self, split: list[dict[str, Any]]) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for sample in split:
            samples.append(self.build_sample(sample))
        return samples

    @abstractmethod
    def build_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def apply_transforms(
        self,
        images: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[Any, Any]:
        if self.sample_transform is not None:
            return self.sample_transform(images, target)

        if self.image_transform is not None:
            images = self.image_transform(images)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return images, target

    def resolve_mask(self, raw_mask: Any) -> torch.Tensor:
        if raw_mask is None:
            if self.masking_mode is MaskingMode.RANDOM and self.mask_patterns is not None:
                index = int(torch.randint(len(self.mask_patterns), size=(1,)).item())
                return self.mask_patterns[index].clone()

            if self.mask_patterns is not None:
                return torch.ones(self.mask_patterns.shape[-1], dtype=torch.bool)

            raise ValueError("raw_mask is None but no default mask configuration is available")

        return torch.as_tensor(raw_mask, dtype=torch.bool)
