from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from mimose.datasets import IMFuseDataset, MaskingMode
from mimose.datasets.imfuse import MASK_PATTERNS
from mimose.enums import TransformKind
from mimose.losses.config import LossConfig
from mimose.losses.m3ae import m3ae_reconstruction_loss
from mimose.models.config import ModelConfig
from mimose.training.config import OptimizerConfig, SchedulerConfig
from mimose.training.trainers.base_trainer import BaseTrainer
from mimose.training.transforms import build_transform_manager
from mimose.training.transforms.base_transforms import TransformManager

DEFAULT_PATCH_SIZE = 128


class M3AETrainer(BaseTrainer):
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
        validation_every: int = 1,
        batch_size: int | None = None,
        num_workers: int | None = None,
        fp16: bool = False,
        compile: bool = False,
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
            int(trainer_kwargs["iter_per_epoch"]) if trainer_kwargs.get("iter_per_epoch") is not None else None
        )
        self.patch_size = int(trainer_kwargs.get("patch_size", DEFAULT_PATCH_SIZE))
        self.debug = bool(trainer_kwargs.get("debug", False))
        self.transform_kind = TransformKind(trainer_kwargs.get("transform_kind", TransformKind.IMFUSE))
        self.train_masking_mode = MaskingMode(trainer_kwargs.get("train_masking_mode", MaskingMode.RANDOM))
        self.val_masking_mode = MaskingMode(trainer_kwargs.get("val_masking_mode", MaskingMode.VALIDATION))
        self.best_val_dice = float("-inf")
        self._train_iterator: Any | None = None

        # Legacy M3AE fine-tuning uses deep supervision plus a cross-view
        # consistency term: two independently-masked views of the same crop are
        # forwarded, both supervised (deep-supervised Dice+CE), and their coarse
        # ``out4`` logits are pulled together by an MSE (legacy weight_kl,
        # feature_level=2). ``consistency_weight`` == 0 keeps deep supervision
        # but drops the second view/MSE; ``deep_supervised`` == False falls back
        # to the plain single-output Dice+CE.
        self.deep_supervised = bool(trainer_kwargs.get("deep_supervised", True))
        self.consistency_weight = float(trainer_kwargs.get("consistency_weight", 1.0))

        # Two-stage schedule matching legacy M3AE's separate pretrain.py (MAE-style
        # reconstruction, ``limage`` trained at its own higher LR) and train.py
        # (segmentation fine-tuning, ``limage`` frozen). Disabled by default
        # (pretrain_fraction=0.0) to keep existing single-stage configs unchanged.
        self.pretrain_fraction = float(trainer_kwargs.get("pretrain_fraction", 0.0))
        if not 0.0 <= self.pretrain_fraction <= 1.0:
            raise ValueError("pretrain_fraction must be within [0, 1]")
        self.pretrain_epochs = round(self.pretrain_fraction * num_epochs)
        self.limage_lr = float(trainer_kwargs.get("limage_lr", 0.005))
        self.limage_reg_weight = float(trainer_kwargs.get("limage_reg_weight", 0.005))
        self._in_pretrain: bool | None = None

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
            validation_every=validation_every,
            batch_size=effective_batch_size,
            num_workers=effective_num_workers,
            fp16=fp16,
            compile=compile,
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

        if self.pretrain is not None and self.resume is None:
            self._load_pretrain()

    def _build_optimizer(self, model: torch.nn.Module | None = None) -> torch.optim.Optimizer:
        if self.pretrain_epochs == 0:
            return super()._build_optimizer(model)

        if self.optimizer_config is None:
            raise RuntimeError("optimizer_config must be set before building an optimizer")

        target_model = model or self._model_for_state()
        if target_model is None:
            raise RuntimeError("model must be initialized before building an optimizer")

        limage_params = [target_model.limage]
        other_params = [
            param for name, param in target_model.named_parameters() if name != "limage"
        ]
        self.optimizer = self.optimizer_config.optim_class(
            [
                {"params": other_params},
                {"params": limage_params, "lr": self.limage_lr},
            ],
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
        )
        return self.optimizer

    def _apply_training_stage(self, epoch: int) -> None:
        """Resume-safe pretrain/finetune stage switch, mirroring
        ``LCKDTrainer._apply_training_phase``: stage membership is a pure
        function of ``epoch``, so it's always correctly recomputed on resume
        without needing to persist any stage state in the checkpoint."""
        if self.pretrain_epochs == 0:
            return

        in_pretrain = epoch < self.pretrain_epochs
        if in_pretrain == self._in_pretrain:
            return

        entering_finetune = self._in_pretrain is True and not in_pretrain
        self._in_pretrain = in_pretrain
        model = self._model_for_state()
        if model is not None:
            model.limage.requires_grad_(in_pretrain)

        if entering_finetune and self.scheduler_config is not None:
            # Legacy runs two independent CosineAnnealingLR schedules (one per
            # script/stage); reproduce that by restarting the anneal over the
            # remaining epochs once fine-tuning begins, rather than letting a
            # single schedule decay continuously across the stage boundary.
            remaining_epochs = self.num_epochs - self.pretrain_epochs
            previous_t_max = self.scheduler_config.kwargs.get("T_max")
            self.scheduler_config.kwargs["T_max"] = remaining_epochs
            self._build_scheduler(self.optimizer)
            if previous_t_max is not None:
                self.scheduler_config.kwargs["T_max"] = previous_t_max

    def train_epoch(self, epoch: int) -> dict[str, float]:
        if self.train_loader is None:
            raise RuntimeError("train_loader must be initialized before training")

        self._apply_training_stage(epoch)

        self.model.train()
        steps = self.iter_per_epoch if self.iter_per_epoch is not None else len(self.train_loader)
        in_pretrain = bool(self._in_pretrain)
        totals = (
            {"loss": 0.0, "recon": 0.0, "smoothness": 0.0}
            if in_pretrain
            else {"loss": 0.0, "cross": 0.0, "dice": 0.0, "consistency": 0.0}
        )
        stage_label = "Pretrain" if in_pretrain else "Train"

        iterations = 0
        with self._progress(disable=not self.is_main_process) as progress:
            task_id = progress.add_task(
                f"{stage_label} {epoch + 1}/{self.num_epochs}",
                total=steps,
                metrics="",
                vram=self._vram_text(),
            )
            for _ in range(steps):
                batch = self._next_train_batch()
                metrics = self._pretrain_step(batch) if in_pretrain else self._train_step(batch)
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

        if self._in_pretrain:
            return self._pretrain_val_epoch(epoch)

        self.model.eval()
        loss_sum = 0.0
        wt_sum = 0.0
        tc_sum = 0.0
        et_sum = 0.0
        etpp_sum = 0.0
        wt_hd95_sum = 0.0
        tc_hd95_sum = 0.0
        et_hd95_sum = 0.0
        etpp_hd95_sum = 0.0
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
                    prediction = pred.argmax(dim=1)
                    target_labels = seg.squeeze(1)
                    wt, tc, et, etpp = self._evaluate_scores(prediction, target_labels)
                    wt_hd95, tc_hd95, et_hd95, etpp_hd95 = self._evaluate_hd95(prediction, target_labels)

                    batch_size = images.shape[0]
                    sample_count += batch_size
                    loss_sum += float(seg_loss.item()) * batch_size
                    wt_sum += float(wt.sum().item())
                    tc_sum += float(tc.sum().item())
                    et_sum += float(et.sum().item())
                    etpp_sum += float(etpp.sum().item())
                    wt_hd95_sum += float(wt_hd95.sum().item())
                    tc_hd95_sum += float(tc_hd95.sum().item())
                    et_hd95_sum += float(et_hd95.sum().item())
                    etpp_hd95_sum += float(etpp_hd95.sum().item())
                    progress.update(
                        task_id,
                        advance=1,
                        vram=self._vram_text(),
                        metrics=f"loss {float(seg_loss.item()):.4f}  avg {loss_sum / max(sample_count, 1):.4f}",
                    )
                    if self.debug:
                        break

        self.model.train()

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
            "wt_hd95": wt_hd95_sum / divisor,
            "tc_hd95": tc_hd95_sum / divisor,
            "et_hd95": et_hd95_sum / divisor,
            "etpp_hd95": etpp_hd95_sum / divisor,
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
            }
        )
        return payload

    def _center_crop_to_patch(self, images: torch.Tensor) -> torch.Tensor:
        """The pretraining reconstruction head (unlike ``predict``) requires
        inputs at exactly ``patch_size`` in every spatial dimension, but the
        val split's "test" transform leaves images at their native
        (uncropped) size. Deterministically center-crop to match, since
        pretrain validation only needs a representative patch, not full
        spatial coverage."""
        patch = self.patch_size
        spatial_shape = images.shape[-3:]
        if tuple(spatial_shape) == (patch, patch, patch):
            return images
        if any(dim < patch for dim in spatial_shape):
            raise RuntimeError(
                f"M3AE pretrain validation requires spatial dimensions of at least "
                f"{(patch, patch, patch)}, got {tuple(spatial_shape)}"
            )
        starts = [(dim - patch) // 2 for dim in spatial_shape]
        h0, w0, d0 = starts
        return images[:, :, h0 : h0 + patch, w0 : w0 + patch, d0 : d0 + patch]

    def _pretrain_val_epoch(self, epoch: int) -> dict[str, float]:
        assert self.val_loader is not None

        self.model.eval()
        loss_sum = 0.0
        sample_count = 0

        with torch.no_grad():
            with self._progress(disable=not self.is_main_process) as progress:
                task_id = progress.add_task(
                    f"Pretrain val {epoch + 1}/{self.num_epochs}",
                    total=len(self.val_loader),
                    metrics="",
                    vram=self._vram_text(),
                )
                for batch in self.val_loader:
                    images = batch["images"].to(self.device, non_blocking=True)
                    mask = batch["mask"].to(self.device, non_blocking=True).bool()
                    images = self._center_crop_to_patch(images)

                    model = self._model_for_state()
                    with self._autocast_context():
                        recon = model.reconstruct(images, mask)
                        metrics = m3ae_reconstruction_loss(
                            recon, images, model.limage, reg_weight=self.limage_reg_weight
                        )

                    batch_size = images.shape[0]
                    sample_count += batch_size
                    loss_sum += float(metrics["loss"].item()) * batch_size
                    progress.update(
                        task_id,
                        advance=1,
                        vram=self._vram_text(),
                        metrics=f"loss {float(metrics['loss'].item()):.4f}  avg {loss_sum / max(sample_count, 1):.4f}",
                    )
                    if self.debug:
                        break

        self.model.train()
        return {"loss": loss_sum / max(sample_count, 1)}

    def _pretrain_step(self, batch: dict[str, Any]) -> dict[str, float]:
        images = batch["images"].to(self.device, non_blocking=True)
        mask = batch["mask"].to(self.device, non_blocking=True).bool()
        with self._autocast_context():
            # Go through self.model(...) (not the DDP-unwrapped module) so
            # DistributedDataParallel's gradient-sync hooks stay attached for
            # the subsequent backward pass; see M3AE.forward's docstring.
            recon = self.model(images, mask, mode="reconstruct")
            limage = self._model_for_state().limage
            metrics = m3ae_reconstruction_loss(recon, images, limage, reg_weight=self.limage_reg_weight)
            loss = metrics["loss"]

        self._backward_step(loss)
        return {
            "loss": float(loss.item()),
            "recon": float(metrics["recon"].item()),
            "smoothness": float(metrics["smoothness"].item()),
        }

    def _random_view_mask(self, reference_mask: torch.Tensor) -> torch.Tensor:
        """Sample a fresh per-sample missing-modality mask for the second view,
        drawn from the same 15-pattern pool the RANDOM dataset masking uses."""
        indices = torch.randint(len(MASK_PATTERNS), size=(reference_mask.shape[0],))
        return MASK_PATTERNS[indices].to(device=reference_mask.device, dtype=reference_mask.dtype)

    def _train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        images = batch["images"].to(self.device, non_blocking=True)
        seg = batch["seg"].to(self.device, non_blocking=True).long()
        mask = batch["mask"].to(self.device, non_blocking=True).bool()
        with self._autocast_context():
            target = self._seg_to_one_hot(seg)

            if not self.deep_supervised:
                output = self.model(images, mask)
                metrics = self._loss_impl().training_loss(output, target)
                loss = metrics["loss"]
                cross, dice = metrics["cross"], metrics["dice"]
                consistency = images.new_tensor(0.0)
            else:
                loss_impl = self._loss_impl()
                seg_outputs, out4_logits = self.model(images, mask, mode="segment_train")
                first = loss_impl.deep_supervised_training_loss(seg_outputs, target)
                loss, cross, dice = first["loss"], first["cross"], first["dice"]
                consistency = images.new_tensor(0.0)

                if self.consistency_weight > 0:
                    # Second independently-masked view of the same crop; both
                    # views are supervised (legacy trains over the doubled
                    # batch) and their out4 logits are matched by MSE.
                    view2_mask = self._random_view_mask(mask)
                    seg_outputs2, out4_logits2 = self.model(images, view2_mask, mode="segment_train")
                    second = loss_impl.deep_supervised_training_loss(seg_outputs2, target)
                    loss = loss + second["loss"]
                    cross = cross + second["cross"]
                    dice = dice + second["dice"]
                    consistency = F.mse_loss(out4_logits, out4_logits2)
                    loss = loss + (self.consistency_weight * consistency)

        self._backward_step(loss)
        return {
            "loss": float(loss.item()),
            "cross": float(cross.item()),
            "dice": float(dice.item()),
            "consistency": float(consistency.item()),
        }
