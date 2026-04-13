from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
import os
from pathlib import Path

from typing import Any

import click
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from brainchmark.losses.config import LossConfig
from brainchmark.models.config import ModelConfig
from brainchmark.training.config import OptimizerConfig, SchedulerConfig


LOGGER = logging.getLogger(__name__)


class BaseTrainer(ABC):
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
        dataset_type:str|None = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.model_config = model_config
        self.loss_config = loss_config
        self.custom_trainer_kwargs = custom_trainer_kwargs or {}
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.num_epochs = int(num_epochs)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.seed = seed
        self.pretrain = Path(pretrain) if pretrain is not None else None
        self.wandb_project = wandb_project
        self.wandb_mode = wandb_mode
        self.split_file: Path | None = None
        self.train_split: list[dict[str, Any]] | None = None
        self.val_split: list[dict[str, Any]] | None = None
        self.test_split: list[dict[str, Any]] | None = None

        self.distributed = self._should_use_distributed()
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = self._resolve_device()
        self.dataset_type = dataset_type

        self.model: torch.nn.Module | None = None
        self.loss_fn: Any | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any | None = None
        self.wandb_run: Any | None = None

        self.current_epoch = 0
        self.best_val_loss = float("inf")

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.resume = self._resolve_resume_path(resume)
        self._owns_process_group = False
        self._setup_distributed()
        self.model = self._build_model()
        self.wrap_model_for_distributed()
        self._build_loss()
        self._build_optimizer()
        self._build_scheduler()
        self._load_dataset_splits()
        self.train_set, self.val_set = self.build_datasets()
        self.train_loader, self.val_loader = self.build_dataloaders(batch_size=batch_size,
                                                                   num_workers=num_workers)

    def fit(self) -> None:

        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()

        start_epoch = 0
        if self.resume:
            start_epoch = self.load_checkpoint(self.resume)

        self._init_wandb()

        try:
            for epoch in range(start_epoch, self.num_epochs):
                self.current_epoch = epoch
                self._set_loader_epoch(self.train_loader, epoch)
                self._set_loader_epoch(self.val_loader, epoch)

                train_metrics = self._reduce_metrics(self.train_epoch(epoch))

                val_metrics = self._reduce_metrics(self.val_epoch(epoch))
                self._step_scheduler(val_metrics)
                self._log_wandb_epoch(epoch, train_metrics, val_metrics)
                self.save_checkpoint(epoch, is_best=self._is_best_checkpoint(val_metrics))
        finally:
            self._finish_wandb()
            self._cleanup_distributed()

    @abstractmethod
    def train_epoch(self, epoch: int) -> dict[str, float]:
        pass

    @abstractmethod
    def val_epoch(self, epoch: int) -> dict[str, float]:
        pass

    @abstractmethod
    def build_datasets(self) -> tuple[Dataset, Dataset]:
        pass

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
        }


        last_path = self.checkpoint_dir / "model_last.pth"
        torch.save(checkpoint, last_path)

        epoch_path = self.checkpoint_dir / f"model_{epoch}.pth"
        torch.save(checkpoint, epoch_path)
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)

        return last_path

    def _build_model(self) -> torch.nn.Module:
        if self.model_config is None:
            raise RuntimeError("model_config must be set before building a model")

        model = self.model_config.model_class(**self.model_config.kwargs)
        return model.to(self.device)

    def _build_optimizer(
        self,
        model: torch.nn.Module | None = None,
    ) -> torch.optim.Optimizer:
        if self.optimizer_config is None:
            raise RuntimeError("optimizer_config must be set before building an optimizer")

        target_model = model or self._model_for_state()
        if target_model is None:
            raise RuntimeError("model must be initialized before building an optimizer")

        optimizer_kwargs: dict[str, Any] = {
            "lr": self.optimizer_config.lr,
            "weight_decay": self.optimizer_config.weight_decay,
        }
        if self.optimizer_config.betas is not None:
            optimizer_kwargs["betas"] = self.optimizer_config.betas
        if self.optimizer_config.momentum is not None:
            optimizer_kwargs["momentum"] = self.optimizer_config.momentum
        if self.optimizer_config.eps is not None:
            optimizer_kwargs["eps"] = self.optimizer_config.eps

        self.optimizer = self.optimizer_config.optim_class(
            target_model.parameters(),
            **optimizer_kwargs,
        )
        return self.optimizer

    def _build_loss(self) -> Any | None:
        if self.loss_config is None:
            self.loss_fn = None
            return None

        self.loss_fn = self.loss_config.loss_class(**self.loss_config.kwargs)
        return self.loss_fn

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> Any | None:
        if self.scheduler_config is None:
            self.scheduler = None
            return None

        target_optimizer = optimizer or self.optimizer
        if target_optimizer is None:
            raise RuntimeError(
                "optimizer must be initialized before building a scheduler"
            )

        self.scheduler = self.scheduler_config.scheduler_class(
            target_optimizer,
            **self.scheduler_config.kwargs,
        )
        return self.scheduler

    def _load_dataset_splits(self) -> None:
        dataset_type = self.dataset_type
        split_file = self.custom_trainer_kwargs.get("split_file")

        if dataset_type is None and split_file is None:
            return

        dataset_key = str(dataset_type).lower() if dataset_type is not None else None
        resolved_split_file = self._resolve_split_file(split_file)
        split_payload = json.loads(resolved_split_file.read_text())
        if dataset_key is None or dataset_key not in split_payload:
            raise RuntimeError(
                f"Dataset split '{dataset_key}' not found in {resolved_split_file}"
            )

        dataset_splits = split_payload[dataset_key]
        self.split_file = resolved_split_file
        self.train_split = list(dataset_splits.get("train", []))
        self.val_split = list(dataset_splits.get("val", []))
        self.test_split = list(dataset_splits.get("test", []))

    def _resolve_split_file(self, split_file: str | Path | None) -> Path:
        splits_dir = Path(__file__).resolve().parents[2] / "data" / "splits"
        if split_file is None:
            return splits_dir / "split.json"

        candidate = Path(split_file)
        if candidate.is_absolute() or candidate.exists():
            return candidate

        resolved = splits_dir / candidate
        if resolved.exists():
            return resolved

        raise FileNotFoundError(
            "split file not found. Use an absolute path, a valid relative path, "
            f"or a filename under {splits_dir}: {split_file}"
        )

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def wrap_model_for_distributed(self) -> torch.nn.Module:
        if self.model is None:
            raise RuntimeError("model must be initialized before DDP wrapping")
        if not self.distributed or isinstance(self.model, DistributedDataParallel):
            return self.model
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
        )

        return self.model

    def build_dataloaders(
        self,
        *,
        batch_size: int,
        num_workers: int ,
    ) -> tuple[DataLoader, DataLoader]:
        if self.train_set is None or self.val_set is None:
            raise RuntimeError("Remember to implement your build_datasets method")
        train_sampler, val_sampler = None, None
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
        train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            num_workers=num_workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
        )
        val_loader = DataLoader(
            self.val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            sampler=val_sampler,
            drop_last=False,
        )
        return train_loader, val_loader

    def _resolve_resume_path(self, resume: bool) -> Path | None:
        if not resume:
            return None
        checkpoint_path = self.checkpoint_dir / "model_last.pth"
        if not checkpoint_path.is_file():
            raise click.ClickException(
                f"resume checkpoint not found at {checkpoint_path}"
            )
        return checkpoint_path


    def _resolve_device(self) -> torch.device:
        if not torch.cuda.is_available():
            click.ClickException("Torch couldn't find any cuda device!")
        if self.distributed and torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        return torch.device("cuda")

    def _step_scheduler(self, val_metrics: dict[str, float]) -> None:
        if self.scheduler is None:
            return

        if self.scheduler_config is not None and self.scheduler_config.scheduler_class is ReduceLROnPlateau:
            if "loss" not in val_metrics:
                raise RuntimeError("plateau scheduler requires validation loss")
            self.scheduler.step(val_metrics["loss"])
        else:
            self.scheduler.step()

    def _init_wandb(self) -> None:
        if not self.is_main_process:
            return
        if self.wandb_project is None:
            return
        if self.wandb_mode == "disabled":
            return

        try:
            import wandb
        except ImportError as exc:
            raise click.ClickException(
                "wandb logging requested but the 'wandb' package is not installed"
            ) from exc

        self.wandb_run = wandb.init(
            project=self.wandb_project,
            dir=str(self.output_dir),
            mode=self.wandb_mode or "online",
            config=self._wandb_config_payload(),
        )

    def _finish_wandb(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None

    def _log_wandb_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        if self.wandb_run is None:
            return

        payload: dict[str, Any] = {"epoch": epoch + 1}
        payload.update({f"train/{key}": value for key, value in train_metrics.items()})
        payload.update({f"val/{key}": value for key, value in val_metrics.items()})

        current_lr = self._current_lr()
        if current_lr is not None:
            payload["lr"] = current_lr

        self.wandb_run.log(payload)

    def _wandb_config_payload(self) -> dict[str, Any]:
        config: dict[str, Any] = {"input_dir": str(self.input_dir), "output_dir": str(self.output_dir),
                                  "num_epochs": self.num_epochs, "batch_size": self.batch_size,
                                  "num_workers": self.num_workers, "seed": self.seed,
                                  "train_transforms": self.train_transforms, "test_transforms": self.test_transforms,
                                  "model_class": self.model_config.model_class.__name__,
                                  "model_kwargs": dict(self.model_config.kwargs)}
        if self.optimizer_config is not None:
            config["optimizer_class"] = self.optimizer_config.optim_class.__name__
            config["optimizer"] = {
                "lr": self.optimizer_config.lr,
                "weight_decay": self.optimizer_config.weight_decay,
                "betas": self.optimizer_config.betas,
                "momentum": self.optimizer_config.momentum,
                "eps": self.optimizer_config.eps,
            }
        if self.loss_config is not None:
            config["loss_class"] = self.loss_config.loss_class.__name__
            config["loss_kwargs"] = dict(self.loss_config.kwargs)
        if self.scheduler_config is not None:
            config["scheduler_class"] = self.scheduler_config.scheduler_class.__name__
            config["scheduler_kwargs"] = dict(self.scheduler_config.kwargs)
        if self.pretrain is not None:
            config["pretrain"] = str(self.pretrain)
        if self.custom_trainer_kwargs:
            config["custom_trainer_kwargs"] = dict(self.custom_trainer_kwargs)
        return config

    def _current_lr(self) -> float | None:
        if self.optimizer is None or not self.optimizer.param_groups:
            return None
        return float(self.optimizer.param_groups[0]["lr"])

    def _is_best_checkpoint(self, val_metrics: dict[str, float]) -> bool:
        if "loss" not in val_metrics:
            return False

        val_loss = float(val_metrics["loss"])
        if val_loss >= self.best_val_loss:
            return False

        self.best_val_loss = val_loss
        return True


    def _setup_distributed(self) -> None:
        if not self.distributed or dist.is_initialized():
            return

        backend = "nccl"
        if self.device.type == "cuda":
            torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
        self._owns_process_group = True

    def _cleanup_distributed(self) -> None:
        if self.distributed and dist.is_initialized() and self._owns_process_group:
            dist.destroy_process_group()

    @staticmethod
    def _should_use_distributed() -> bool:
        return (
            dist.is_available()
            and int(os.environ.get("WORLD_SIZE", "1")) > 1
            and "RANK" in os.environ
        )

    def _barrier(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def _model_for_state(self) -> torch.nn.Module | None:
        if self.model is None:
            return None
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model

    def _reduce_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if not self.distributed or not metrics:
            return metrics

        reduced: dict[str, float] = {}
        for key, value in metrics.items():
            tensor = torch.tensor(float(value), device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reduced[key] = tensor.item() / self.world_size
        return reduced

    @staticmethod
    def _set_loader_epoch(loader: Any, epoch: int) -> None:
        if loader is None:
            return

        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
            return

        batch_sampler = getattr(loader, "batch_sampler", None)
        nested_sampler = getattr(batch_sampler, "sampler", None)
        if nested_sampler is not None and hasattr(nested_sampler, "set_epoch"):
            nested_sampler.set_epoch(epoch)
