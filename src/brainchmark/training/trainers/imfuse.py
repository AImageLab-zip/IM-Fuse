from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from brainchmark.datasets import DatasetType, IMFuseDataset, MaskingMode
from brainchmark.losses.config import LossConfig
from brainchmark.models.config import ModelConfig
from brainchmark.training.config import OptimizerConfig, SchedulerConfig
from brainchmark.training.trainers.base_trainer import BaseTrainer


LOGGER = logging.getLogger(__name__)
DEFAULT_PATCH_SIZE = 128


class IMFuseTrainer(BaseTrainer):
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        custom_trainer_kwargs: dict[str, Any] | None = None,
        model_config: ModelConfig | None = None,
        loss_config: LossConfig | None = None,
        optimizer_config: OptimizerConfig | None = None,
        scheduler_config: SchedulerConfig | None = None,
        train_transforms: str | None = None,
        test_transforms: str | None = None,
        num_epochs: int = 1,
        batch_size: int | None = None,
        num_workers: int | None = None,
        resume: bool = False,
        seed: int | None = None,
        pretrain: str | Path | None = None,
        wandb_project: str | None = None,
        wandb_mode: str | None = None,
        dataset_type:str|None = None
    ) -> None:
        trainer_kwargs = dict(custom_trainer_kwargs or {})
        self.dataset_type = self._resolve_dataset_type(
             dataset_type
        )
        self.dataname = str(
            trainer_kwargs.get(
                "dataname",
                "BRATS2023" if self.dataset_type is DatasetType.BRATS23 else "BRATS2018",
            )
        )
        self.iter_per_epoch = (
            int(trainer_kwargs["iter_per_epoch"])
            if trainer_kwargs.get("iter_per_epoch") is not None
            else None
        )
        self.region_fusion_start_epoch = int(
            trainer_kwargs.get("region_fusion_start_epoch", 0)
        )
        self.patch_size = int(trainer_kwargs.get("patch_size", DEFAULT_PATCH_SIZE))
        self.debug = bool(trainer_kwargs.get("debug", False))
        self.best_val_dice = float("-inf")
        self._train_iterator: Any | None = None

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
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            num_epochs=num_epochs,
            batch_size=effective_batch_size,
            num_workers=effective_num_workers,
            resume=resume,
            seed=seed,
            pretrain=pretrain,
            wandb_project=wandb_project,
            wandb_mode=wandb_mode,
            dataset_type=dataset_type
        )

        if self.pretrain is not None and self.resume is None:
            self._load_pretrain()

    @property
    def num_classes(self) -> int:
        if self.model_config is not None:
            explicit = self.model_config.kwargs.get("num_cls")
            if explicit is not None:
                return int(explicit)
        return 4

    def train_epoch(self, epoch: int) -> dict[str, float]:
        if self.train_loader is None:
            raise RuntimeError("train_loader must be initialized before training")

        self.model.train()
        self._set_aux_training_flag(True)
        steps = self.iter_per_epoch or len(self.train_loader)
        totals = {
            "loss": 0.0,
            "fusecross": 0.0,
            "fusedice": 0.0,
            "sepcross": 0.0,
            "sepdice": 0.0,
            "prmcross": 0.0,
            "prmdice": 0.0,
        }

        iterations = 0
        for step in range(steps):
            batch = self._next_train_batch()
            metrics = self._train_step(batch, epoch)
            iterations += 1
            for key, value in metrics.items():
                totals[key] += value
            LOGGER.info(
                "Epoch %s/%s Iter %s/%s Loss %.4f fusecross:%.4f fusedice:%.4f "
                "sepcross:%.4f sepdice:%.4f prmcross:%.4f prmdice:%.4f",
                epoch + 1,
                self.num_epochs,
                step + 1,
                steps,
                metrics["loss"],
                metrics["fusecross"],
                metrics["fusedice"],
                metrics["sepcross"],
                metrics["sepdice"],
                metrics["prmcross"],
                metrics["prmdice"],
            )
            if self.debug:
                break

        return {
            key: value / max(iterations, 1)
            for key, value in totals.items()
        }

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
            for batch in self.val_loader:
                images = batch["images"].to(self.device, non_blocking=True)
                seg = batch["seg"].to(self.device, non_blocking=True).long()
                mask = batch["mask"].to(self.device, non_blocking=True).bool()

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

        train_set = IMFuseDataset(
            root=self.input_dir,
            masking_mode=MaskingMode.RANDOM,
            split=self.train_split,
        )
        val_set = IMFuseDataset(
            root=self.input_dir,
            masking_mode=MaskingMode.VALIDATION,
            split=self.val_split,
        )
        return train_set, val_set

    def build_dataloaders(
        self,
        *,
        batch_size: int,
        num_workers: int,
    ) -> tuple[DataLoader, DataLoader]:
        train_sampler: DistributedSampler | None = None
        val_sampler: DistributedSampler | None = None
        if self.distributed:
            train_sampler = DistributedSampler(
                self.train_set,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            val_sampler = DistributedSampler(
                self.val_set,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )

        pin_memory = self.device.type == "cuda"
        train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=train_sampler,
            drop_last=True,
        )
        val_loader = DataLoader(
            self.val_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            sampler=val_sampler,
            drop_last=False,
        )
        return train_loader, val_loader

    def _build_model(self) -> torch.nn.Module:
        if self.model_config is None:
            raise RuntimeError("model_config must be set before building a model")

        model_kwargs = dict(self.model_config.kwargs)
        model_kwargs.setdefault("num_cls", self.num_classes)
        model_class = self.model_config.model_class
        model = model_class(**model_kwargs)
        return model.to(self.device)

    def _build_loss(self) -> Any | None:
        if self.loss_config is None:
            return super()._build_loss()

        loss_kwargs = dict(self.loss_config.kwargs)
        loss_kwargs.setdefault("num_classes", self.num_classes)
        self.loss_fn = self.loss_config.loss_class(**loss_kwargs)
        return self.loss_fn

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        model = self._model_for_state()
        if model is None:
            raise RuntimeError("model must be initialized before loading a checkpoint")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["state_dict"])
        if self.optimizer is not None and checkpoint.get("optim_dict") is not None:
            self.optimizer.load_state_dict(checkpoint["optim_dict"])
        if self.scheduler is not None and checkpoint.get("scheduler_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_dict"])

        self.best_val_dice = float(
            checkpoint.get(
                "best_val_dice",
                checkpoint.get("val_Dice_best", self.best_val_dice),
            )
        )
        self.best_val_loss = float(checkpoint.get("best_val_loss", self.best_val_loss))
        epoch = int(checkpoint.get("epoch", -1)) + 1
        LOGGER.info("resumed from %s at epoch %s", checkpoint_path, epoch)
        return epoch

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> Path:
        model = self._model_for_state()
        if model is None:
            raise RuntimeError("model must be initialized before saving a checkpoint")
        if not self.is_main_process:
            return self.checkpoint_dir / "model_last.pth"

        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optim_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_val_loss": self.best_val_loss,
            "best_val_dice": self.best_val_dice,
        }
        last_path = self.checkpoint_dir / "model_last.pth"
        torch.save(checkpoint, last_path)

        epoch_path = self.checkpoint_dir / f"model_{epoch}.pth"
        torch.save(checkpoint, epoch_path)
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
        return last_path

    def _is_best_checkpoint(self, val_metrics: dict[str, float]) -> bool:
        if "dice" not in val_metrics:
            return False

        val_dice = float(val_metrics["dice"])
        if val_dice <= self.best_val_dice:
            return False

        self.best_val_dice = val_dice
        return True

    def _wandb_config_payload(self) -> dict[str, Any]:
        payload = super()._wandb_config_payload()
        payload.update(
            {
                "dataset_type": self.dataset_type,
                "iter_per_epoch": self.iter_per_epoch,
                "region_fusion_start_epoch": self.region_fusion_start_epoch,
                "patch_size": self.patch_size,
                "debug": self.debug,
                "split_file": str(self.split_file),
            }
        )
        return payload

    @staticmethod
    def _resolve_dataset_type(
        dataset_type: Any | None,
    ) -> DatasetType:

        return DatasetType(str(dataset_type).lower())


    def _load_pretrain(self) -> None:
        checkpoint = torch.load(self.pretrain, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model = self._model_for_state()
        if model is None:
            raise RuntimeError("model must be initialized before loading pretrained weights")
        model.load_state_dict(state_dict, strict=False)
        LOGGER.info("loaded pretrained weights from %s", self.pretrain)

    def _next_train_batch(self) -> dict[str, Any]:
        if self._train_iterator is None:
            self._train_iterator = iter(self.train_loader)

        try:
            return next(self._train_iterator)
        except StopIteration:
            self._train_iterator = iter(self.train_loader)
            return next(self._train_iterator)

    def _train_step(self, batch: dict[str, Any], epoch: int) -> dict[str, float]:
        images = batch["images"].to(self.device, non_blocking=True)
        seg = batch["seg"].to(self.device, non_blocking=True).long()
        mask = batch["mask"].to(self.device, non_blocking=True).bool()
        images, seg = self._crop_pair(images, seg, random_crop=True)
        target = self._seg_to_one_hot(seg)

        fuse_pred, sep_preds, prm_preds = self.model(images, mask)
        metrics = self._loss_impl().training_loss(
            fuse_pred,
            sep_preds,
            prm_preds,
            target,
            include_fuse=epoch >= self.region_fusion_start_epoch,
        )
        loss = metrics["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "fusecross": float(metrics["fusecross"].item()),
            "fusedice": float(metrics["fusedice"].item()),
            "sepcross": float(metrics["sepcross"].item()),
            "sepdice": float(metrics["sepdice"].item()),
            "prmcross": float(metrics["prmcross"].item()),
            "prmdice": float(metrics["prmdice"].item()),
        }

    def _predict_volume(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        _, _, height, width, depth = images.shape
        if (height, width, depth) == (self.patch_size, self.patch_size, self.patch_size):
            return self.model(images, mask)

        h_starts = self._window_starts(height)
        w_starts = self._window_starts(width)
        d_starts = self._window_starts(depth)
        prediction = torch.zeros(
            images.size(0),
            self.num_classes,
            height,
            width,
            depth,
            device=self.device,
        )
        weight = torch.zeros(
            images.size(0),
            1,
            height,
            width,
            depth,
            device=self.device,
        )

        for h in h_starts:
            for w in w_starts:
                for d in d_starts:
                    patch = images[
                        :,
                        :,
                        h : h + self.patch_size,
                        w : w + self.patch_size,
                        d : d + self.patch_size,
                    ]
                    patch_pred = self.model(patch, mask)
                    prediction[
                        :,
                        :,
                        h : h + self.patch_size,
                        w : w + self.patch_size,
                        d : d + self.patch_size,
                    ] += patch_pred
                    weight[
                        :,
                        :,
                        h : h + self.patch_size,
                        w : w + self.patch_size,
                        d : d + self.patch_size,
                    ] += 1
        return prediction / weight

    def _crop_pair(
        self,
        images: torch.Tensor,
        seg: torch.Tensor,
        *,
        random_crop: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width, depth = images.shape
        if (height, width, depth) == (self.patch_size, self.patch_size, self.patch_size):
            return images, seg

        if min(height, width, depth) < self.patch_size:
            raise RuntimeError(
                "IMFuseTrainer expects preprocessed crops to be at least "
                f"{self.patch_size} voxels along each spatial dimension, got "
                f"{(height, width, depth)}"
            )

        start_h = self._crop_start(height, self.patch_size, random_crop)
        start_w = self._crop_start(width, self.patch_size, random_crop)
        start_d = self._crop_start(depth, self.patch_size, random_crop)
        images = images[
            :,
            :,
            start_h : start_h + self.patch_size,
            start_w : start_w + self.patch_size,
            start_d : start_d + self.patch_size,
        ]
        seg = seg[
            :,
            :,
            start_h : start_h + self.patch_size,
            start_w : start_w + self.patch_size,
            start_d : start_d + self.patch_size,
        ]
        return images, seg

    def _seg_to_one_hot(self, seg: torch.Tensor) -> torch.Tensor:
        labels = seg.squeeze(1).long()
        one_hot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes)
        return one_hot.permute(0, 4, 1, 2, 3).float()

    def _segmentation_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss_impl().segmentation_loss(pred, target)

    def _evaluate_scores(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.num_classes != 4:
            raise RuntimeError("IMFuseTrainer currently supports 4-class BraTS labels only")

        eps = 1e-8
        o1 = (output == 1).float()
        t1 = (target == 1).float()
        o2 = (output == 2).float()
        t2 = (target == 2).float()
        o3 = (output == 3).float()
        t3 = (target == 3).float()

        o3_post = torch.where(
            o3.sum(dim=(1, 2, 3), keepdim=True) < 500,
            torch.zeros_like(o3),
            o3,
        )

        whole_pred = o1 + o2 + o3
        whole_target = t1 + t2 + t3
        core_pred = o1 + o3
        core_target = t1 + t3

        wt = self._dice_from_binary(whole_pred, whole_target, eps)
        tc = self._dice_from_binary(core_pred, core_target, eps)
        et = self._dice_from_binary(o3, t3, eps)
        etpp = self._dice_from_binary(o3_post, t3, eps)
        return wt, tc, et, etpp

    @staticmethod
    def _dice_from_binary(
        pred: torch.Tensor,
        target: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        intersection = 2 * (pred * target).sum(dim=(1, 2, 3)) + eps
        denominator = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
        return intersection / denominator

    def _set_aux_training_flag(self, enabled: bool) -> None:
        model = self._model_for_state()
        if model is not None and hasattr(model, "is_training"):
            model.is_training = enabled

    @staticmethod
    def _crop_start(size: int, patch_size: int, random_crop: bool) -> int:
        if size == patch_size:
            return 0
        max_start = size - patch_size
        if not random_crop:
            return max_start // 2
        return int(torch.randint(max_start + 1, size=(1,)).item())

    def _window_starts(self, size: int) -> list[int]:
        if size <= self.patch_size:
            return [0]

        stride = self.patch_size // 2
        starts = list(range(0, max(size - self.patch_size, 0), stride))
        last_start = size - self.patch_size
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def _loss_impl(self) -> Any:
        if self.loss_fn is None:
            raise RuntimeError("loss_config must be set before computing losses")
        return self.loss_fn
