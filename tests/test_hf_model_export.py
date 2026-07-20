from __future__ import annotations

import json
from pathlib import Path

import torch

from mimose.models.abstract_model import AbstractModel


class DummyHFModel(AbstractModel):
    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.width = width
        self.layer = torch.nn.Linear(width, width)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return images

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return images


def test_abstract_model_export_hf_pretrained_writes_config_and_weights(
    tmp_path: Path,
) -> None:
    model = DummyHFModel(width=4)
    model._mimose_model_kwargs = {"width": 4}
    model._mimose_model_name = "DummyHFModel"

    export_dir = model.export_hf_pretrained(tmp_path / "hf")

    config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))

    assert config["model_class"] == "DummyHFModel"
    assert config["model_kwargs"] == {"width": 4}
    assert (export_dir / "final_weights_only.safetensors").is_file()
