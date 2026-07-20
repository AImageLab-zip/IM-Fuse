from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from mimose.datasets import IMFuseDataset, MaskingMode
from mimose.enums import TransformKind
from mimose.losses.config import LossConfig
from mimose.models.config import ModelConfig
from mimose.models.many_mimosas import ALL_MASK_PATTERNS
from mimose.training.config import OptimizerConfig, SchedulerConfig
from mimose.training.trainers.base_trainer import CONSOLE, BaseTrainer
from mimose.training.transforms import build_transform_manager
from mimose.training.transforms.base_transforms import TransformManager


class ManyMimosasKDTrainer(BaseTrainer):
    """Two-phase trainer for ``ManyMimosasKD``: epochs
    ``[0, teacher_pretrain_epochs)`` train only the embedded teacher
    (direct segmentation loss, full modalities); at ``teacher_pretrain_epochs``
    the teacher is frozen and the distillation phase begins, running every
    one of the 15 missing-modality patterns per batch (like
    ``ManyMimosasTrainer``) with a single cached, no-grad teacher forward
    per batch supplying the hint/logit-KD targets -- so each training step
    costs 1 extra teacher forward (frozen, no backward) plus the same 15
    student forward/backward passes ``ManyMimosasTrainer`` already does.

    ``teacher_pretrain_fraction``/``teacher_lr`` mirror
    ``MambaVitAKDTrainer``'s identically-named knobs (see that class's
    docstring for the full rationale behind the phase split and per-
    param-group teacher LR).
    """

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
        wandb_entity: str | None = None,
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
        self.debug = bool(trainer_kwargs.get("debug", False))
        self.transform_kind = TransformKind(
            trainer_kwargs.get("transform_kind", TransformKind.TINYMIMOSA)
        )
        self.best_val_dice = float("-inf")
        self._train_iterator: Any | None = None
        self.mask_patterns = ALL_MASK_PATTERNS.clone()

        self.teacher_pretrain_fraction = float(trainer_kwargs.get("teacher_pretrain_fraction", 0.0))
        if not 0.0 <= self.teacher_pretrain_fraction <= 1.0:
            raise ValueError(
                "teacher_pretrain_fraction must be within [0, 1], got "
                f"{self.teacher_pretrain_fraction}"
            )
        self.teacher_pretrain_epochs = round(self.teacher_pretrain_fraction * num_epochs)
        raw_teacher_lr = trainer_kwargs.get("teacher_lr")
        self.teacher_lr = float(raw_teacher_lr) if raw_teacher_lr is not None else None
        self._in_teacher_pretrain: bool | None = None

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
            wandb_entity=wandb_entity,
            wandb_mode=wandb_mode,
            wandb_run_name=wandb_run_name,
            dataset_type=dataset_type,
            push_to_hf=push_to_hf,
            hf_repo=hf_repo,
        )
        self.dataset_type = self._resolve_dataset_type(dataset_type)
        self.mask_patterns = self.mask_patterns.to(self.device)

        if self.pretrain is not None and self.resume is None:
            self._load_pretrain()

    def _build_optimizer(self, model: torch.nn.Module | None = None) -> torch.optim.Optimizer:
        if self.teacher_lr is None or self.teacher_pretrain_epochs == 0:
            return super()._build_optimizer(model)

        if self.optimizer_config is None:
            raise RuntimeError("optimizer_config must be set before building an optimizer")

        target_model = model or self._model_for_state()
        if target_model is None:
            raise RuntimeError("model must be initialized before building an optimizer")

        teacher_params = list(target_model.teacher.parameters())
        other_params = [
            param for name, param in target_model.named_parameters()
            if not name.startswith("teacher.")
        ]
        self.optimizer = self.optimizer_config.optim_class(
            [
                {"params": other_params},
                {"params": teacher_params, "lr": self.teacher_lr},
            ],
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
        )
        return self.optimizer

    def _build_scheduler(self, optimizer: torch.optim.Optimizer | None = None) -> Any | None:
        if self.scheduler_config is None or self.teacher_pretrain_epochs == 0:
            return super()._build_scheduler(optimizer)

        if self._in_teacher_pretrain is not None:
            return super()._build_scheduler(optimizer)

        target_optimizer = optimizer or self.optimizer
        if target_optimizer is None:
            raise RuntimeError("optimizer must be initialized before building a scheduler")

        phase_kwargs = dict(self.scheduler_config.kwargs)
        for duration_key in ("total_iters", "T_max"):
            if duration_key in phase_kwargs:
                phase_kwargs[duration_key] = self.teacher_pretrain_epochs
        self.scheduler = self.scheduler_config.scheduler_class(target_optimizer, **phase_kwargs)
        return self.scheduler

    def _apply_training_stage(self, epoch: int) -> None:
        if self.teacher_pretrain_epochs == 0:
            return

        in_teacher_pretrain = epoch < self.teacher_pretrain_epochs
        if in_teacher_pretrain == self._in_teacher_pretrain:
            return

        entering_distillation = (not in_teacher_pretrain) and (self._in_teacher_pretrain is not False)
        self._in_teacher_pretrain = in_teacher_pretrain

        if entering_distillation:
            model = self._model_for_state()
            if model is not None:
                model.teacher.requires_grad_(False)
                model.teacher.eval()

            if self.scheduler_config is not None:
                remaining_epochs = self.num_epochs - self.teacher_pretrain_epochs
                previous = {}
                for duration_key in ("total_iters", "T_max"):
                    if duration_key in self.scheduler_config.kwargs:
                        previous[duration_key] = self.scheduler_config.kwargs[duration_key]
                        self.scheduler_config.kwargs[duration_key] = remaining_epochs
                self._build_scheduler(self.optimizer)
                self.scheduler_config.kwargs.update(previous)

        if self.is_main_process:
            phase_name = "distillation" if entering_distillation else "teacher pretrain"
            CONSOLE.print(
                f"[bold cyan]ManyMimosasKD[/bold cyan] entering {phase_name} phase at epoch "
                f"{epoch + 1}/{self.num_epochs} "
                f"(teacher_pretrain_epochs={self.teacher_pretrain_epochs}, "
                f"teacher_pretrain_fraction={self.teacher_pretrain_fraction})"
            )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        if self.train_loader is None:
            raise RuntimeError("train_loader must be initialized before training")

        self._apply_training_stage(epoch)
        self.model.train()
        in_teacher_pretrain = bool(self._in_teacher_pretrain)
        self._set_aux_training_flag(True)

        steps = self.iter_per_epoch if self.iter_per_epoch is not None else len(self.train_loader)
        base_keys = ("loss", "cross", "dice")
        extra_keys = () if in_teacher_pretrain else ("hint", "kd")
        totals = {key: 0.0 for key in base_keys + extra_keys}
        stage_label = "Teacher pretrain" if in_teacher_pretrain else "Distill"

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
                metrics = (
                    self._teacher_pretrain_step(batch)
                    if in_teacher_pretrain
                    else self._train_step(batch)
                )
                iterations += 1
                for key, value in metrics.items():
                    totals[key] += value
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

        return {key: value / max(iterations, 1) for key, value in totals.items()}

    def val_epoch(self, epoch: int) -> dict[str, float]:
        if self.val_loader is None:
            raise RuntimeError("val_loader must be initialized before validation")

        if self._in_teacher_pretrain:
            return self._teacher_pretrain_val_epoch(epoch)

        self.model.eval()
        self._set_aux_training_flag(False)
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
        num_patterns = self.mask_patterns.size(0)

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
                    batch_size = images.shape[0]
                    target_labels = seg.squeeze(1)
                    target = self._seg_to_one_hot(seg)

                    combo_loss = 0.0
                    combo_wt = 0.0
                    combo_tc = 0.0
                    combo_et = 0.0
                    combo_etpp = 0.0
                    combo_wt_hd95 = 0.0
                    combo_tc_hd95 = 0.0
                    combo_et_hd95 = 0.0
                    combo_etpp_hd95 = 0.0

                    for pattern in self.mask_patterns:
                        mask = pattern.unsqueeze(0).expand(batch_size, -1)
                        with self._autocast_context():
                            pred = self._predict_volume(images, mask)
                            seg_loss = self._segmentation_loss(pred, target)
                        prediction = pred.argmax(dim=1)
                        wt, tc, et, etpp = self._evaluate_scores(prediction, target_labels)
                        wt_hd95, tc_hd95, et_hd95, etpp_hd95 = self._evaluate_hd95(
                            prediction, target_labels
                        )
                        combo_loss += float(seg_loss.item())
                        combo_wt += float(wt.sum().item())
                        combo_tc += float(tc.sum().item())
                        combo_et += float(et.sum().item())
                        combo_etpp += float(etpp.sum().item())
                        combo_wt_hd95 += float(wt_hd95.sum().item())
                        combo_tc_hd95 += float(tc_hd95.sum().item())
                        combo_et_hd95 += float(et_hd95.sum().item())
                        combo_etpp_hd95 += float(etpp_hd95.sum().item())

                    sample_count += batch_size
                    loss_sum += (combo_loss / num_patterns) * batch_size
                    wt_sum += combo_wt / num_patterns
                    tc_sum += combo_tc / num_patterns
                    et_sum += combo_et / num_patterns
                    etpp_sum += combo_etpp / num_patterns
                    wt_hd95_sum += combo_wt_hd95 / num_patterns
                    tc_hd95_sum += combo_tc_hd95 / num_patterns
                    et_hd95_sum += combo_et_hd95 / num_patterns
                    etpp_hd95_sum += combo_etpp_hd95 / num_patterns
                    progress.update(
                        task_id,
                        advance=1,
                        vram=self._vram_text(),
                        metrics=(
                            f"loss {combo_loss / num_patterns:.4f}  "
                            f"avg {loss_sum / max(sample_count, 1):.4f}"
                        ),
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
            "wt_hd95": wt_hd95_sum / divisor,
            "tc_hd95": tc_hd95_sum / divisor,
            "et_hd95": et_hd95_sum / divisor,
            "etpp_hd95": etpp_hd95_sum / divisor,
        }

    def _teacher_pretrain_val_epoch(self, epoch: int) -> dict[str, float]:
        assert self.val_loader is not None

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
                    f"Teacher val {epoch + 1}/{self.num_epochs}",
                    total=len(self.val_loader),
                    metrics="",
                    vram=self._vram_text(),
                )
                for batch in self.val_loader:
                    images = batch["images"].to(self.device, non_blocking=True)
                    seg = batch["seg"].to(self.device, non_blocking=True).long()

                    model = self._model_for_state()
                    with self._autocast_context():
                        pred = model.predict_teacher(images)
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
                        metrics=(
                            f"loss {float(seg_loss.item()):.4f}  "
                            f"avg {loss_sum / max(sample_count, 1):.4f}"
                        ),
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
            )
        # Every mask pattern (including the teacher's own full-modality one)
        # is generated internally each step, so both splits always load
        # every modality unmasked -- mirrors ManyMimosasTrainer.
        train_set = IMFuseDataset(
            root=self.input_dir,
            masking_mode=MaskingMode.FULL,
            split=self.train_split,
            sample_transform=partial(transform_manager, mode="train"),
        )
        val_set = IMFuseDataset(
            root=self.input_dir,
            masking_mode=MaskingMode.FULL,
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
                "debug": self.debug,
                "num_mask_patterns": int(self.mask_patterns.size(0)),
                "split_file": str(self.split_file),
                "teacher_pretrain_fraction": self.teacher_pretrain_fraction,
                "teacher_pretrain_epochs": self.teacher_pretrain_epochs,
                "teacher_lr": self.teacher_lr,
            }
        )
        return payload

    def _teacher_pretrain_step(self, batch: dict[str, Any]) -> dict[str, float]:
        images = batch["images"].to(self.device, non_blocking=True)
        seg = batch["seg"].to(self.device, non_blocking=True).long()
        mask = batch["mask"].to(self.device, non_blocking=True).bool()
        with self._autocast_context():
            target = self._seg_to_one_hot(seg)
            teacher_out = self.model(images, mask, mode="teacher_pretrain")
            metrics = self._loss_impl().teacher_pretrain_loss(teacher_out, target)
            loss = metrics["loss"]

        self._backward_step(loss)
        return {
            "loss": float(loss.item()),
            "cross": float(metrics["cross"].item()),
            "dice": float(metrics["dice"].item()),
        }

    def _train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        images = batch["images"].to(self.device, non_blocking=True)
        seg = batch["seg"].to(self.device, non_blocking=True).long()
        batch_size = images.shape[0]
        target = self._seg_to_one_hot(seg)

        full_mask = torch.ones(batch_size, images.size(1), dtype=torch.bool, device=self.device)
        with torch.no_grad(), self._autocast_context():
            teacher_hints = self.model(images, full_mask, mode="teacher_forward")

        totals = {"loss": 0.0, "cross": 0.0, "dice": 0.0, "hint": 0.0, "kd": 0.0}
        for pattern in self.mask_patterns:
            mask = pattern.unsqueeze(0).expand(batch_size, -1)
            with self._autocast_context():
                outputs = self.model(images, mask, teacher_hints=teacher_hints)
                metrics = self._loss_impl().training_loss(outputs, target)
                loss = metrics["loss"]

            self._backward_step(loss)
            totals["loss"] += float(loss.item())
            totals["cross"] += float(metrics["cross"].item())
            totals["dice"] += float(metrics["dice"].item())
            totals["hint"] += float(metrics["hint"].item())
            totals["kd"] += float(metrics["kd"].item())

        num_patterns = self.mask_patterns.size(0)
        return {key: value / num_patterns for key, value in totals.items()}


__all__ = ["ManyMimosasKDTrainer"]
