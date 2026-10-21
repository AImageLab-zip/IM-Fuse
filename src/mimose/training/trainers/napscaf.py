from __future__ import annotations

from typing import Any

import torch

from mimose.training.trainers.mstkdnet import MSTKDTrainer

DEFAULT_SLICES_PER_PATCH = 16
SLICE_SAMPLING_MODES = ("tumor", "random")


class NAPSCAFTrainer(MSTKDTrainer):
    """Trains the 2D NAP-SCAF network on axial slices of MiMoSe's 3D patches.

    Data loading, modality masking, validation (slice-wise ``predict`` over the
    whole volume) and checkpointing are MST-KDNet's. Each training step draws
    ``slices_per_patch`` axial slices from every patch -- by default only
    tumor-containing ones while there are enough, mirroring legacy's
    preprocessing, which keeps only tumor slices -- and trains on them as one
    batch of 2D images that share their patch's modality mask.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        trainer_kwargs = dict(kwargs.get("custom_trainer_kwargs") or {})
        self.slices_per_patch = int(trainer_kwargs.get("slices_per_patch", DEFAULT_SLICES_PER_PATCH))
        if self.slices_per_patch < 1:
            raise ValueError("slices_per_patch must be >= 1")
        self.slice_sampling = str(trainer_kwargs.get("slice_sampling", "tumor"))
        if self.slice_sampling not in SLICE_SAMPLING_MODES:
            raise ValueError(f"slice_sampling must be one of {SLICE_SAMPLING_MODES}, got {self.slice_sampling!r}")
        super().__init__(*args, **kwargs)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        if self.train_loader is None:
            raise RuntimeError("train_loader must be initialized before training")

        self.model.train()
        self._set_aux_training_flag(True)
        steps = self.iter_per_epoch if self.iter_per_epoch is not None else len(self.train_loader)
        totals: dict[str, float] = {}

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
                metrics = self._train_step(batch, epoch)
                iterations += 1
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
                progress.update(
                    task_id,
                    advance=1,
                    vram=self._vram_text(),
                    metrics=(
                        f"loss {metrics['loss']:.4f}  "
                        f"avg {totals['loss'] / iterations:.4f}"
                    ),
                )

                if self.debug:
                    break

        return {
            key: value / max(iterations, 1)
            for key, value in totals.items()
        }

    def _sample_slice_indices(self, seg: torch.Tensor) -> torch.Tensor:
        """Axial (last-dim) slice indices for one [H, W, D] label patch."""
        depth = seg.size(-1)
        count = min(self.slices_per_patch, depth)
        order = torch.randperm(depth, device=seg.device)
        if self.slice_sampling == "tumor":
            has_tumor = (seg > 0).flatten(0, 1).any(dim=0)[order]
            # Tumor slices first (in random order), then the rest as filler.
            order = torch.cat([order[has_tumor], order[~has_tumor]])
        return order[:count]

    def _slice_batch(
        self, images: torch.Tensor, seg: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[B, 4, H, W, D] patches -> [N, 4, H, W] slices, [N, H, W] labels, [N, 4] masks."""
        image_slices, label_slices, mask_slices = [], [], []
        for sample_images, sample_seg, sample_mask in zip(images, seg, mask):
            labels = sample_seg[0]
            indices = self._sample_slice_indices(labels)
            image_slices.append(sample_images[..., indices].permute(3, 0, 1, 2))
            label_slices.append(labels[..., indices].permute(2, 0, 1))
            mask_slices.append(sample_mask.expand(indices.numel(), -1))
        return torch.cat(image_slices), torch.cat(label_slices), torch.cat(mask_slices)

    def _wandb_config_payload(self) -> dict[str, Any]:
        payload = super()._wandb_config_payload()
        payload.update({"slices_per_patch": self.slices_per_patch, "slice_sampling": self.slice_sampling})
        return payload

    def _train_step(self, batch: dict[str, Any], epoch: int) -> dict[str, float]:
        images = batch["images"].to(self.device, non_blocking=True)
        seg = batch["seg"].to(self.device, non_blocking=True).long()
        mask = batch["mask"].to(self.device, non_blocking=True).bool()
        slices, labels, slice_mask = self._slice_batch(images, seg, mask)
        with self._autocast_context():
            logits = self.model(slices, slice_mask)
            metrics = self._loss_impl().training_loss(logits, labels)
            loss = metrics["loss"]

        self._backward_step(loss)
        return {key: float(value.item()) for key, value in metrics.items()}
