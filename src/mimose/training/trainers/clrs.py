from __future__ import annotations

from dataclasses import replace
from functools import partial
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import PolynomialLR

from mimose.datasets import IMFuseDataset, MaskingMode
from mimose.enums import TransformKind
from mimose.losses.config import LossConfig
from mimose.models.clrs import num_modals
from mimose.models.config import ModelConfig
from mimose.training.config import OptimizerConfig, SchedulerConfig
from mimose.training.trainers.base_trainer import BaseTrainer
from mimose.training.transforms import build_transform_manager
from mimose.training.transforms.base_transforms import TransformManager


DEFAULT_PATCH_SIZE = 80
SYNTHESIS_CHANNELS = 64
SYNTHESIS_EMBED_DIM = 64

LOGGER = logging.getLogger(__name__)


def _synthesis_checkpoint_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_stem(checkpoint_path.stem + "_synthesis")


class _Residual3DBlock(nn.Module):
    """Port of ``legacy/CLRS/networks/synthesis.py::Residual3DBlock``."""

    def __init__(self, in_channels: int, out_channels: int, p_dim: int) -> None:
        super().__init__()
        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels + p_dim, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor, p_i: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        p_expanded = p_i.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, x.size(2), x.size(3), x.size(4))
        out = self.bn2(self.conv2(torch.cat([out, p_expanded], dim=1)))
        return F.relu(out + residual)


class _SynthesisEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, p_dim: int) -> None:
        super().__init__()
        self.stage1 = _Residual3DBlock(in_channels, base_channels, p_dim)
        self.downsample1 = nn.Conv3d(base_channels, base_channels * 2, kernel_size=2, stride=2)
        self.stage2 = _Residual3DBlock(base_channels * 2, base_channels * 2, p_dim)
        self.downsample2 = nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=2, stride=2)
        self.stage3 = _Residual3DBlock(base_channels * 4, base_channels * 4, p_dim)
        self.downsample3 = nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, p_i: torch.Tensor) -> list[torch.Tensor]:
        s1 = self.stage1(x, p_i)
        d1 = self.downsample1(s1)
        s2 = self.stage2(d1, p_i)
        d2 = self.downsample2(s2)
        s3 = self.stage3(d2, p_i)
        d3 = self.downsample3(s3)
        return [s1, s2, s3, d3]


class _SynthesisMirrorDecoder(nn.Module):
    def __init__(self, base_channels: int, p_dim: int) -> None:
        super().__init__()
        self.upsample0 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.stage0 = _Residual3DBlock(base_channels * 4, base_channels * 4, p_dim)

        self.upsample1 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.stage1 = _Residual3DBlock(base_channels * 4, base_channels * 2, p_dim)

        self.upsample2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.stage2 = _Residual3DBlock(base_channels * 2, base_channels, p_dim)

        self.stage3 = _Residual3DBlock(base_channels * 2, base_channels, p_dim)
        self.final_conv = nn.Conv3d(base_channels, base_channels, kernel_size=1)

    def forward(self, features: list[torch.Tensor], p_i: torch.Tensor) -> torch.Tensor:
        s1, s2, s3, d3 = features

        u0 = self.stage0(self.upsample0(d3), p_i)
        u1 = self.stage1(self.upsample1(torch.cat([u0, s3], dim=1)), p_i)
        u2 = self.stage2(self.upsample2(torch.cat([u1, s2], dim=1)), p_i)
        u3 = self.stage3(torch.cat([u2, s1], dim=1), p_i)
        return self.final_conv(u3)


class _GModel(nn.Module):
    """Port of ``legacy/CLRS/networks/synthesis.py::G_Model``: a per-modality
    conditioned residual encoder-decoder used to reconstruct one modality's
    bottleneck slice from another's, as an auxiliary cyclic-consistency
    regularizer."""

    def __init__(
        self,
        in_channels: int = SYNTHESIS_CHANNELS,
        base_channels: int = SYNTHESIS_CHANNELS,
        p_dim: int = SYNTHESIS_EMBED_DIM,
    ) -> None:
        super().__init__()
        self.encoder = _SynthesisEncoder(in_channels, base_channels, p_dim)
        self.decoder = _SynthesisMirrorDecoder(base_channels, p_dim)
        self.modality_embedding = nn.Parameter(torch.randn(p_dim))

    def forward(self, x_i: torch.Tensor) -> torch.Tensor:
        p_i = self.modality_embedding.unsqueeze(0).expand(x_i.size(0), -1)
        features = self.encoder(x_i, p_i)
        return torch.tanh(self.decoder(features, p_i))


class CLRSTrainer(BaseTrainer):
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        custom_trainer_kwargs: dict[str, Any] | None = None,
        model_config: ModelConfig | None = None,
        loss_config: LossConfig | None = None,
        optimizer_config: OptimizerConfig | None = None,
        scheduler_config: SchedulerConfig | None = None,
        transform_manager: TransformManager | None = None,
        num_epochs: int = 1,
        batch_size: int | None = None,
        num_workers: int | None = None,
        fp16: bool = False,
        resume: bool = False,
        try_resume: bool = False,
        seed: int | None = None,
        pretrain: str | Path | None = None,
        wandb_project: str | None = None,
        wandb_mode: str | None = None,
        wandb_run_name: str | None = None,
        dataset_type: str | None = None,
        push_to_hf: bool = False,
        hf_repo: str | None = None,
    ) -> None:
        trainer_kwargs = dict(custom_trainer_kwargs or {})
        self.dataname = str(trainer_kwargs.get("dataname", "BRATS2023"))
        self.iter_per_epoch = (
            int(trainer_kwargs["iter_per_epoch"])
            if trainer_kwargs.get("iter_per_epoch") is not None
            else None
        )
        self.patch_size = int(trainer_kwargs.get("patch_size", DEFAULT_PATCH_SIZE))
        self.debug = bool(trainer_kwargs.get("debug", False))
        self.transform_kind = TransformKind(
            trainer_kwargs.get("transform_kind", TransformKind.IMFUSE)
        )
        self.train_masking_mode = MaskingMode(
            trainer_kwargs.get("train_masking_mode", MaskingMode.RANDOM)
        )
        self.val_masking_mode = MaskingMode(
            trainer_kwargs.get("val_masking_mode", MaskingMode.VALIDATION)
        )
        self.use_cyclic_synthesis = bool(trainer_kwargs.get("use_cyclic_synthesis", True))
        self.synthesis_lr = float(trainer_kwargs.get("synthesis_lr", 1e-4))
        self.best_val_dice = float("-inf")
        self._train_iterator: Any | None = None

        if (
            scheduler_config is not None
            and scheduler_config.scheduler_class is PolynomialLR
        ):
            scheduler_config = replace(
                scheduler_config,
                kwargs={**scheduler_config.kwargs, "total_iters": num_epochs},
            )

        effective_batch_size = 1 if batch_size is None else batch_size
        effective_num_workers = 0 if num_workers is None else num_workers
        super().__init__(
            input_dir=input_dir,
            output_dir=output_dir,
            custom_trainer_kwargs=trainer_kwargs,
            model_config=model_config,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            transform_manager=transform_manager,
            num_epochs=num_epochs,
            batch_size=effective_batch_size,
            num_workers=effective_num_workers,
            fp16=fp16,
            resume=resume,
            try_resume=try_resume,
            seed=seed,
            pretrain=pretrain,
            wandb_project=wandb_project,
            wandb_mode=wandb_mode,
            wandb_run_name=wandb_run_name,
            dataset_type=dataset_type,
            push_to_hf=push_to_hf,
            hf_repo=hf_repo,
        )
        self.dataset_type = self._resolve_dataset_type(dataset_type)

        self.synthesis_generators: nn.ModuleList | None = None
        self.synthesis_optimizer: torch.optim.Optimizer | None = None
        if self.use_cyclic_synthesis:
            self.synthesis_generators = nn.ModuleList(
                [_GModel() for _ in range(num_modals)]
            ).to(self.device)
            self.synthesis_optimizer = torch.optim.Adam(
                self.synthesis_generators.parameters(), lr=self.synthesis_lr
            )

        if self.pretrain is not None and self.resume is None:
            self._load_pretrain()

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> Path:
        last_path = super().save_checkpoint(epoch, is_best)
        if self.use_cyclic_synthesis and self.is_main_process:
            assert self.synthesis_generators is not None
            assert self.synthesis_optimizer is not None
            synthesis_checkpoint = {
                "epoch": epoch,
                "state_dict": self.synthesis_generators.state_dict(),
                "optim_dict": self.synthesis_optimizer.state_dict(),
            }
            torch.save(synthesis_checkpoint, _synthesis_checkpoint_path(last_path))
            if is_best:
                best_path = self.checkpoint_dir / "best.pth"
                torch.save(synthesis_checkpoint, _synthesis_checkpoint_path(best_path))
        return last_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        start_epoch = super().load_checkpoint(checkpoint_path)
        if self.use_cyclic_synthesis:
            synthesis_path = _synthesis_checkpoint_path(Path(checkpoint_path))
            if synthesis_path.is_file():
                assert self.synthesis_generators is not None
                assert self.synthesis_optimizer is not None
                synthesis_checkpoint = torch.load(synthesis_path, map_location=self.device)
                self.synthesis_generators.load_state_dict(synthesis_checkpoint["state_dict"])
                self.synthesis_optimizer.load_state_dict(synthesis_checkpoint["optim_dict"])
            else:
                LOGGER.warning(
                    "Resuming CLRS training with cyclic synthesis enabled, but no "
                    "companion synthesis checkpoint was found at %s — the synthesis "
                    "generators will restart from their initial weights.",
                    synthesis_path,
                )
        return start_epoch

    def train_epoch(self, epoch: int) -> dict[str, float]:
        if self.train_loader is None:
            raise RuntimeError("train_loader must be initialized before training")

        self.model.train()
        self._set_aux_training_flag(True)
        if self.synthesis_generators is not None:
            self.synthesis_generators.train()
        steps = self.iter_per_epoch if self.iter_per_epoch is not None else len(self.train_loader)
        totals = {
            "loss": 0.0,
            "fusecross": 0.0,
            "fusedice": 0.0,
            "specmodalloss": 0.0,
            "speclogitclsloss": 0.0,
            "synthesisloss": 0.0,
        }

        iterations = 0
        with self._progress(disable=not self.is_main_process) as progress:
            task_id = progress.add_task(
                f"Train {epoch + 1}/{self.num_epochs}",
                total=steps,
                metrics="",
                vram=self._vram_text(),
            )
            for _ in range(steps):
                batch = self._next_train_batch()
                metrics = self._train_step(batch)
                iterations += 1
                for key, value in metrics.items():
                    totals[key] += value
                progress.update(
                    task_id,
                    advance=1,
                    vram=self._vram_text(),
                    metrics=f"loss {metrics['loss']:.4f}  avg {totals['loss'] / iterations:.4f}",
                )
                if self.debug:
                    break

        return {key: value / max(iterations, 1) for key, value in totals.items()}

    def val_epoch(self, epoch: int) -> dict[str, float]:
        if self.val_loader is None:
            raise RuntimeError("val_loader must be initialized before validation")

        self.model.eval()
        self._set_aux_training_flag(False)
        loss_sum = 0.0
        wt_sum = 0.0
        tc_sum = 0.0
        et_sum = 0.0
        etpp_sum = 0.0
        sample_count = 0

        with torch.no_grad():
            with self._progress(disable=not self.is_main_process) as progress:
                task_id = progress.add_task(
                    f"Val {epoch + 1}/{self.num_epochs}",
                    total=len(self.val_loader),
                    metrics="",
                    vram=self._vram_text(),
                )
                for batch in self.val_loader:
                    images = batch["images"].to(self.device, non_blocking=True)
                    seg = batch["seg"].to(self.device, non_blocking=True).long()
                    mask = batch["mask"].to(self.device, non_blocking=True).bool()

                    with self._autocast_context():
                        pred = self._predict_volume(images, mask)
                        target = self._seg_to_one_hot(seg)
                        seg_loss = self._segmentation_loss(pred, target)
                    wt, tc, et, etpp = self._evaluate_scores(pred.argmax(dim=1), seg.squeeze(1))

                    batch_size = images.shape[0]
                    sample_count += batch_size
                    loss_sum += float(seg_loss.item()) * batch_size
                    wt_sum += float(wt.sum().item())
                    tc_sum += float(tc.sum().item())
                    et_sum += float(et.sum().item())
                    etpp_sum += float(etpp.sum().item())
                    progress.update(
                        task_id,
                        advance=1,
                        vram=self._vram_text(),
                        metrics=f"loss {float(seg_loss.item()):.4f}  avg {loss_sum / max(sample_count, 1):.4f}",
                    )
                    if self.debug:
                        break

        self.model.train()
        self._set_aux_training_flag(True)

        divisor = max(sample_count, 1)
        wt_score = wt_sum / divisor
        tc_score = tc_sum / divisor
        et_score = et_sum / divisor
        dice_score = (wt_score + tc_score + et_score) / 3.0
        return {
            "loss": loss_sum / divisor,
            "wt": wt_score,
            "tc": tc_score,
            "et": et_score,
            "etpp": etpp_sum / divisor,
            "dice": dice_score,
        }

    def build_datasets(self) -> tuple[Dataset, Dataset]:
        if self.train_split is None or self.val_split is None:
            raise RuntimeError("train_split and val_split must be loaded before building datasets")

        transform_manager = self.transform_manager
        if transform_manager is None:
            transform_manager = build_transform_manager(
                self.transform_kind,
                model_kwargs=(self.model_config.kwargs if self.model_config is not None else None),
                crop_size=(self.patch_size, self.patch_size, self.patch_size),
            )
        train_set = IMFuseDataset(
            root=self.input_dir,
            masking_mode=self.train_masking_mode,
            split=self.train_split,
            sample_transform=partial(transform_manager, mode="train"),
        )
        val_set = IMFuseDataset(
            root=self.input_dir,
            masking_mode=self.val_masking_mode,
            split=self.val_split,
            sample_transform=partial(transform_manager, mode="test"),
        )
        return train_set, val_set

    def _wandb_config_payload(self) -> dict[str, Any]:
        payload = super()._wandb_config_payload()
        payload.update(
            {
                "dataset_type": self.dataset_type,
                "iter_per_epoch": self.iter_per_epoch,
                "patch_size": self.patch_size,
                "debug": self.debug,
                "train_masking_mode": self.train_masking_mode,
                "val_masking_mode": self.val_masking_mode,
                "split_file": str(self.split_file),
                "use_cyclic_synthesis": self.use_cyclic_synthesis,
                "synthesis_lr": self.synthesis_lr,
            }
        )
        return payload

    def _synthesis_loss(self, generated_map: list[torch.Tensor]) -> torch.Tensor:
        """Cycles each modality's (detached) bottleneck slice through its own
        generator to predict the next modality's slice, mirroring
        ``synthesize_modality`` in ``legacy/CLRS/segmentor/model_trainer.py``.
        Inputs are detached so this loss only trains the synthesis
        generators, not the main segmentation network — matching the
        legacy trainer's separate-optimizer design."""
        assert self.synthesis_generators is not None
        slices = [modal_slice.detach() for modal_slice in generated_map]
        num_modality = len(slices)

        current = slices[0]
        noise = torch.randn_like(current)
        input_data = (current + current + noise) / 2.0
        total_loss = current.new_tensor(0.0)
        for modal_index in range(num_modality):
            prediction = self.synthesis_generators[modal_index](input_data)
            target = slices[(modal_index + 1) % num_modality]
            total_loss = total_loss + F.mse_loss(prediction, target)
            input_data = (slices[(modal_index + 1) % num_modality] + prediction) / 2.0

        return total_loss / num_modality

    def _train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        images = batch["images"].to(self.device, non_blocking=True)
        seg = batch["seg"].to(self.device, non_blocking=True).long()
        mask = batch["mask"].to(self.device, non_blocking=True).bool()

        if self.use_cyclic_synthesis:
            assert self.synthesis_optimizer is not None
            self.synthesis_optimizer.zero_grad()
        self.optimizer.zero_grad()

        with self._autocast_context():
            target = self._seg_to_one_hot(seg)
            outputs = self.model(images, mask)
            metrics = self._loss_impl().training_loss(outputs, target, mask=mask)
            loss = metrics["loss"]

            synthesis_loss = loss.new_tensor(0.0)
            if self.use_cyclic_synthesis:
                synthesis_loss = self._synthesis_loss(outputs[3])
                loss = loss + synthesis_loss

        if self.amp_enabled:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            if self.use_cyclic_synthesis:
                self.grad_scaler.step(self.synthesis_optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
            if self.use_cyclic_synthesis:
                self.synthesis_optimizer.step()

        result = {key: float(value.item()) for key, value in metrics.items()}
        result["synthesisloss"] = float(synthesis_loss.item())
        return result
