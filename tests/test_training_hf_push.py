from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

from mimose.models.abstract_model import AbstractModel
from mimose.training.trainers.base_trainer import BaseTrainer


class DummyTrainerModel(AbstractModel):
    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(width, width)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return images

    def predict(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return images


def _build_trainer(tmp_path: Path) -> BaseTrainer:
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.output_dir = tmp_path / "artifacts"
    trainer.wandb_run_name = "imfuse23_training"
    trainer.push_to_hf = True
    trainer.hf_repo = "owner/repo"
    trainer.rank = 0
    trainer.model = DummyTrainerModel(width=4)
    trainer.model._mimose_model_kwargs = {"width": 4}
    trainer.model._mimose_model_name = "DummyTrainerModel"
    return trainer


def test_base_trainer_exports_hf_artifacts_into_run_directory(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    checkpoint_path = tmp_path / "final_weights_only.safetensors"
    checkpoint_path.write_bytes(b"weights")

    export_dir = trainer._export_hf_artifacts(checkpoint_path)
    config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))

    assert export_dir == tmp_path / "artifacts" / "huggingface" / "imfuse23_training"
    assert config["model_kwargs"] == {"width": 4}
    assert (export_dir / "final_weights_only.safetensors").is_file()


def test_base_trainer_uploads_hf_artifacts_into_run_subdirectory(tmp_path: Path) -> None:
    trainer = _build_trainer(tmp_path)
    checkpoint_path = tmp_path / "final_weights_only.safetensors"
    checkpoint_path.write_bytes(b"weights")
    export_dir = trainer._export_hf_artifacts(checkpoint_path)

    captured: dict[str, object] = {}

    class DummyApi:
        def upload_folder(self, **kwargs: object) -> None:
            captured.update(kwargs)

    fake_module = type(sys)("huggingface_hub")
    fake_module.HfApi = lambda: DummyApi()
    sys.modules["huggingface_hub"] = fake_module

    try:
        trainer._upload_hf_artifacts(export_dir)
    finally:
        sys.modules.pop("huggingface_hub", None)

    assert captured["repo_id"] == "owner/repo"
    assert captured["path_in_repo"] == "imfuse23_training"
    assert captured["repo_type"] == "model"


def test_base_trainer_skips_hf_push_on_non_main_process(
    monkeypatch,
    tmp_path: Path,
) -> None:
    trainer = _build_trainer(tmp_path)
    trainer.rank = 1
    checkpoint_path = tmp_path / "final_weights_only.safetensors"
    checkpoint_path.write_bytes(b"weights")

    called = {"value": False}
    monkeypatch.setattr(
        trainer,
        "_upload_hf_artifacts",
        lambda export_dir: called.__setitem__("value", True),
    )

    trainer._maybe_push_to_hf(checkpoint_path)

    assert called["value"] is False


def test_base_trainer_hf_push_warns_on_upload_failure(
    monkeypatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    trainer = _build_trainer(tmp_path)
    checkpoint_path = tmp_path / "final_weights_only.safetensors"
    checkpoint_path.write_bytes(b"weights")

    monkeypatch.setattr(
        trainer,
        "_upload_hf_artifacts",
        lambda export_dir: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with caplog.at_level("WARNING"):
        trainer._maybe_push_to_hf(checkpoint_path)

    assert "Failed to push trained model to Hugging Face" in caplog.text


def test_base_trainer_standalone_push_loads_checkpoint_and_uploads(
    monkeypatch,
    tmp_path: Path,
) -> None:
    trainer = _build_trainer(tmp_path)
    checkpoint_path = tmp_path / "final_weights_only.safetensors"
    save_path = checkpoint_path
    trainer.model.layer.weight.data.fill_(1.0)
    trainer.model.layer.bias.data.fill_(2.0)
    from mimose.checkpoints import save_weights_only_checkpoint

    save_weights_only_checkpoint(trainer.model.state_dict(), save_path)

    trainer.model.layer.weight.data.zero_()
    trainer.model.layer.bias.data.zero_()

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        trainer,
        "_upload_hf_artifacts",
        lambda export_dir: captured.__setitem__("export_dir", export_dir),
    )

    export_dir = trainer.push_checkpoint_to_hf(checkpoint_path)

    assert export_dir == tmp_path / "artifacts" / "huggingface" / "imfuse23_training"
    assert captured["export_dir"] == export_dir
    assert torch.all(trainer.model.layer.weight == 1.0)
    assert torch.all(trainer.model.layer.bias == 2.0)
