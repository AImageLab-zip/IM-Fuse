from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

if TYPE_CHECKING:
    from brainchmark.models.config import ModelConfig
    from brainchmark.training.config import OptimizerConfig, SchedulerConfig


LOGGER = logging.getLogger(__name__)


class BaseTrainer(ABC):
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        trainer: str,
        model_config: ModelConfig | None = None,
        custom_trainer_kwargs: dict[str, Any] | None = None,
        optimizer_config: OptimizerConfig | None = None,
        scheduler_config: SchedulerConfig | None = None,
        train_transforms: str | None = None,
        test_transforms: str | None = None,
        num_epochs: int = 1,
        batch_size: int | None = None,
        num_workers: int | None = None,
        resume: bool | str | Path | None = None,
        seed: int | None = None,
        pretrain: str | Path | None = None,
        wandb_project: str | None = None,
        wandb_mode: str | None = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.trainer = str(trainer)

        self.model_config = model_config
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

        self.distributed = self._should_use_distributed()
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = self._resolve_device()

        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any | None = None
        self.train_loader: Any | None = None
        self.val_loader: Any | None = None

        self.current_epoch = 0
        self.best_val_loss = float("inf")

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.resume = self._resolve_resume_path(resume)
        self._owns_process_group = False

        self._setup_distributed()

    def fit(self) -> None:
        self._configure_logging()
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()

        start_epoch = 0
        if self.resume is not None:
            start_epoch = self.load_checkpoint(self.resume)

        try:
            for epoch in range(start_epoch, self.num_epochs):
                self.current_epoch = epoch
                self._set_loader_epoch(self.train_loader, epoch)
                self._set_loader_epoch(self.val_loader, epoch)

                train_metrics = self._reduce_metrics(self.train_epoch(epoch))
                if self.is_main_process:
                    LOGGER.info("epoch=%s train=%s", epoch + 1, train_metrics)

                val_metrics: dict[str, float] | None = None
                if self.val_loader is not None:
                    val_metrics = self._reduce_metrics(self.val_epoch(epoch))
                    if self.is_main_process:
                        LOGGER.info("epoch=%s val=%s", epoch + 1, val_metrics)

                self._step_scheduler(val_metrics)
                self.save_checkpoint(epoch, is_best=self._is_best_checkpoint(val_metrics))
        finally:
            self._cleanup_distributed()

    @abstractmethod
    def train_epoch(self, epoch: int) -> dict[str, float]:
        pass

    @abstractmethod
    def val_epoch(self, epoch: int) -> dict[str, float]:
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
        self._load_extra_checkpoint_state(checkpoint)

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
        checkpoint.update(self._extra_checkpoint_state())

        last_path = self.checkpoint_dir / "model_last.pth"
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)

        return last_path

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def wrap_model_for_distributed(self) -> torch.nn.Module:
        if self.model is None:
            raise RuntimeError("model must be initialized before DDP wrapping")
        if not self.distributed or isinstance(self.model, DistributedDataParallel):
            return self.model

        if self.device.type == "cuda":
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
        else:
            self.model = DistributedDataParallel(self.model)
        return self.model

    def _configure_logging(self) -> None:
        if logging.getLogger().handlers:
            return
        logging.basicConfig(
            level=logging.INFO if self.is_main_process else logging.WARNING,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    def _resolve_resume_path(self, resume: bool | str | Path | None) -> Path | None:
        if isinstance(resume, bool):
            if not resume:
                return None
            checkpoint_path = self.checkpoint_dir / "model_last.pth"
            if not checkpoint_path.is_file():
                raise RuntimeError(f"resume checkpoint not found at {checkpoint_path}")
            return checkpoint_path

        if resume is None:
            return None
        return Path(resume)

    def _resolve_device(self) -> torch.device:
        if self.distributed and torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _step_scheduler(self, val_metrics: dict[str, float] | None) -> None:
        if self.scheduler is None:
            return

        if val_metrics is not None and "loss" in val_metrics:
            try:
                self.scheduler.step(val_metrics["loss"])
                return
            except TypeError:
                pass

        self.scheduler.step()

    def _is_best_checkpoint(self, val_metrics: dict[str, float] | None) -> bool:
        if val_metrics is None or "loss" not in val_metrics:
            return False

        val_loss = float(val_metrics["loss"])
        if val_loss >= self.best_val_loss:
            return False

        self.best_val_loss = val_loss
        return True

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {}

    def _load_extra_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        return None

    def _setup_distributed(self) -> None:
        if not self.distributed or dist.is_initialized():
            return

        backend = "nccl" if self.device.type == "cuda" else "gloo"
        if self.device.type == "cuda":
            torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
        self._owns_process_group = True

    def _cleanup_distributed(self) -> None:
        if self.distributed and dist.is_initialized() and self._owns_process_group:
            dist.barrier()
            dist.destroy_process_group()

    def _should_use_distributed(self) -> bool:
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

    def _set_loader_epoch(self, loader: Any, epoch: int) -> None:
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
