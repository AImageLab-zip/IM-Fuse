from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from brainchmark.datasets.config import MaskingMode


MASK_PATTERNS = torch.tensor(
    [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, False],
        [False, False, False, True],
        [True, True, False, False],
        [True, False, True, False],
        [True, False, False, True],
        [False, True, True, False],
        [False, True, False, True],
        [False, False, True, True],
        [True, True, True, False],
        [True, True, False, True],
        [True, False, True, True],
        [False, True, True, True],
        [True, True, True, True],
    ],
    dtype=torch.bool,
)


class IMFuseDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        masking_mode: MaskingMode,
        split: list[dict[str, Any]],
        image_transforms: Callable[[torch.Tensor], Any] | None = None,
        target_transforms: Callable[[torch.Tensor], Any] | None = None

    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(f"Dataset directory not found: {self.root}")

        self.image_transform = image_transforms
        self.target_transform = target_transforms
        self.masking_mode = masking_mode
        self.samples = self._load_samples(split)

        if not self.samples:
            raise ValueError(f"No samples found in {self.root}")

    def _load_samples(self, split: list[dict[str, Any]]) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for sample in split:
            subject_name = str(sample["sub"])
            subject_path = self.root / f"{subject_name}.npz"
            if not subject_path.is_file():
                raise ValueError(f"Sample file not found: {subject_path}")
            samples.append(
                {
                    "sub": subject_path,
                    "mask": sample.get("mask"),
                }
            )
        return samples

    def _resolve_mask(self, raw_mask: Any) -> torch.Tensor:
        if raw_mask is None:
            if self.masking_mode is MaskingMode.RANDOM:
                index = int(torch.randint(len(MASK_PATTERNS), size=(1,)).item())
                return MASK_PATTERNS[index].clone()
            return torch.ones(4, dtype=torch.bool)

        return torch.as_tensor(raw_mask, dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with np.load(sample["sub"]) as data:
            images = torch.from_numpy(data["images"]).float()
            seg = torch.from_numpy(data["seg"]).float()

        if self.image_transform is not None:
            images = self.image_transform(images)
        if self.target_transform is not None:
            seg = self.target_transform(seg)

        item: dict[str, Any] = {
            "sub": sample["sub"].stem,
            "images": images,
            "seg": seg,
            "mask": self._resolve_mask(sample["mask"]),
        }
        return item
