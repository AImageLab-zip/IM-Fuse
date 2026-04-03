import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

# Custom imports
from brainchmark.datasets.config import MaskingMode,DatasetType


class IMFuseDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        masking_mode: MaskingMode,
        dataset_type: DatasetType,
        split_file: str | Path | None = None,
        image_transforms: Callable[[torch.Tensor], Any] | None = None,
        target_transforms: Callable[[torch.Tensor], Any] | None = None

    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(f"Dataset directory not found: {self.root}")

        self.image_transform = image_transforms
        self.target_transform = target_transforms
        self.masking_mode = masking_mode
        self.samples = self._load_samples(split_file)

        if not self.samples:
            raise ValueError(f"No samples found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with np.load(sample.path) as data:
            images = torch.from_numpy(data["images"]).float()
            seg = torch.from_numpy(data["seg"]).long()

        if self.image_transform is not None:
            images = self.image_transform(images)
        if self.target_transform is not None:
            seg = self.target_transform(seg)

        item: dict[str, Any] = {
            "case": sample.case,
            "images": images,
            "seg": seg,
            "mask" : sample.mask
        }
        if self.return_mask:
            item["mask"] = sample.mask
        return item




    def _load_json_split(self, split_path: Path) -> list:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        samples: list = []

        if isinstance(payload, list):
            entries = payload
        else:
            entries = []
            for value in payload.values():
                if isinstance(value, list):
                    entries.extend(value)

        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid split entry: {entry}")
            case = str(entry.get("case") or entry.get("sub") or "").strip()
            if not case:
                raise ValueError(f"Missing case/sub field in entry: {entry}")
            samples.append(
                self._build_sample(
                    case=case,
                    mask=self._parse_mask(entry.get("mask")),
                )
            )
        return samples



    def _parse_mask(self, raw_mask: Any) -> list[bool] | None:
        if raw_mask is None or raw_mask == "":
            return None
        if isinstance(raw_mask, list):
            return [bool(value) for value in raw_mask]
        if isinstance(raw_mask, str):
            parsed = ast.literal_eval(raw_mask)
            if not isinstance(parsed, list):
                raise ValueError(f"Mask must parse to a list, got {type(parsed).__name__}")
            return [bool(value) for value in parsed]
        raise ValueError(f"Unsupported mask type: {type(raw_mask).__name__}")
