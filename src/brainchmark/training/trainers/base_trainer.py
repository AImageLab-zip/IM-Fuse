from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from typing import Any

import click
import torch
import torch.distributed as dist
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.nn.parallel import DistributedDataParallel
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from brainchmark.datasets import DatasetType
from brainchmark.training.config import OptimizerConfig, SchedulerConfig
from brainchmark.training.trainers.abstract_trainer import AbstractTrainer
from brainchmark.losses.config import LossConfig
from brainchmark.models.abstract_model import AbstractModel
from brainchmark.models.config import ModelConfig


LOGGER = logging.getLogger(__name__)
CONSOLE = Console()


class BaseTrainer(AbstractTrainer):
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,

        custom_trainer_kwargs: dict[str, Any] | None = None,

        model_config: ModelConfig | None = None,
        loss_config: LossConfig | None = None,
        optimizer_config: OptimizerConfig | None = None,
        scheduler_config: SchedulerConfig | None = None,

        num_epochs: int = 1,
        batch_size: int | None = None,
        num_workers: int | None = None,
        fp16: bool = False,
        resume: bool = False,
        seed: int | None = None,
        pretrain: str | Path | None = None,
        wandb_project: str | None = None,
        wandb_mode: str | None = None,
        wandb_run_name: str | None = None,
        dataset_type:str|None = None,
    ) -> None:
        super().__init__(
            input_dir=input_dir,
            output_dir=output_dir,
            custom_trainer_kwargs=custom_trainer_kwargs,
            model_config=model_config,
            loss_config=loss_config,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            fp16=fp16,
            resume=resume,
            seed=seed,
            pretrain=pretrain,
            wandb_project=wandb_project,
            wandb_mode=wandb_mode,
            wandb_run_name=wandb_run_name,
            dataset_type=dataset_type,
        )

        self.distributed = self._should_use_distributed()
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = self._resolve_device()
        self.best_val_dice = getattr(self, "best_val_dice", float("-inf"))
        self._train_iterator = getattr(self, "_train_iterator", None)
        self.amp_enabled = self.fp16 and self.device.type == "cuda"
        self.grad_scaler = GradScaler("cuda", enabled=self.amp_enabled)
        self.resume = self._resolve_resume_path(self.resume_requested)
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
            self._print_launch_summary()
        self._barrier()

        start_epoch = 0
        if self.resume:
            start_epoch = self.load_checkpoint(self.resume)

        self._init_wandb()

        try:
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
                self.save_final_checkpoint()
            except Exception as exc:
                if self._is_cuda_oom_error(exc):
                    self._handle_cuda_oom()
                raise
        finally:
            self._finish_wandb()
            self._cleanup_distributed()

    def train_epoch(self, epoch: int) -> dict[str, float]:
        raise NotImplementedError

    def val_epoch(self, epoch: int) -> dict[str, float]:
        raise NotImplementedError

    def build_datasets(self) -> tuple[Dataset, Dataset]:
        raise NotImplementedError

    @staticmethod
    def _format_gib(value_bytes: int) -> str:
        return f"{value_bytes / (1024 ** 3):.1f}GiB"

    def _vram_text(self) -> str:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return "cpu"

        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

        used_bytes = torch.cuda.max_memory_allocated(device_index)
        total_bytes = torch.cuda.get_device_properties(device_index).total_memory
        return f"{self._format_gib(used_bytes)}/{self._format_gib(total_bytes)}"

    @property
    def num_classes(self) -> int:
        if self.model_config is not None:
            explicit = self.model_config.kwargs.get("num_cls")
            if explicit is not None:
                return int(explicit)
        return 4

    def _progress(self, *, disable: bool) -> Progress:
        return Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("VRAM {task.fields[vram]}", style="yellow"),
            TextColumn("{task.fields[metrics]}", style="magenta"),
            transient=True,
            disable=disable,
        )

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
            checkpoint.get("best_val_dice", checkpoint.get("val_Dice_best", self.best_val_dice))
        )
        self.best_val_loss = float(checkpoint.get("best_val_loss", self.best_val_loss))
        return int(checkpoint.get("epoch", -1)) + 1

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
        #torch.save(checkpoint, epoch_path)
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)

        return last_path

    def save_final_checkpoint(self) -> Path:
        model = self._model_for_state()
        if model is None:
            raise RuntimeError("model must be initialized before saving a checkpoint")
        if not self.is_main_process:
            return self.checkpoint_dir / "final.pth"

        checkpoint = {
            "epoch": self.current_epoch,
            "state_dict": model.state_dict(),
            "optim_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_val_loss": self.best_val_loss,
        }

        final_path = self.checkpoint_dir / "final.pth"
        torch.save(checkpoint, final_path)
        return final_path

    def _build_model(self) -> torch.nn.Module:
        if self.model_config is None:
            raise RuntimeError("model_config must be set before building a model")

        model_kwargs = dict(self.model_config.kwargs)
        model_kwargs.setdefault("num_cls", self.num_classes)
        model = self.model_config.model_class(**model_kwargs)
        if not isinstance(model, AbstractModel):
            raise RuntimeError(
                f"{self.model_config.model_class.__name__} must inherit from AbstractModel"
            )
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

        loss_kwargs = dict(self.loss_config.kwargs)
        loss_kwargs.setdefault("num_classes", self.num_classes)
        self.loss_fn = self.loss_config.loss_class(**loss_kwargs)
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
            find_unused_parameters=True,
        )

        return self.model

    def build_dataloaders(
        self,
        *,
        batch_size: int,
        num_workers: int,
    ) -> tuple[DataLoader, DataLoader]:
        if self.train_set is None or self.val_set is None:
            raise RuntimeError("Remember to implement your build_datasets method")
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

    @staticmethod
    def _resolve_dataset_type(dataset_type: Any | None) -> DatasetType:
        return DatasetType(str(dataset_type).lower())

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
            raise click.ClickException("Torch couldn't find any cuda device!")
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

        run_id = self._load_saved_wandb_run_id()
        wandb_init_kwargs: dict[str, Any] = {
            "project": self.wandb_project,
            "dir": str(self.output_dir),
            "mode": self.wandb_mode or "online",
            "config": self._wandb_config_payload(),
        }
        wandb_init_kwargs["name"] = self.wandb_run_name
        if self.resume is not None and run_id is not None:
            wandb_init_kwargs["id"] = run_id
            wandb_init_kwargs["resume"] = "must"

        try:
            self.wandb_run = wandb.init(
                **wandb_init_kwargs,
            )
        except Exception as exc:
            message = str(exc).lower()
            auth_markers = (
                "api key",
                "login",
                "not logged",
                "must be logged in",
                "permission denied",
                "unauthorized",
            )
            if any(marker in message for marker in auth_markers):
                raise click.ClickException(
                    "Weights & Biases logging is enabled, but no active W&B login was found.\n\n"
                    "Run `wandb login` in this terminal, then retry.\n"
                    "Or disable W&B with `--wandb-mode disabled`."
                ) from exc
            raise
        self._persist_wandb_run_id()

    @staticmethod
    def _is_cuda_oom_error(exc: Exception) -> bool:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        if isinstance(exc, RuntimeError):
            message = str(exc).lower()
            return "out of memory" in message and "cuda" in message
        return False

    def _handle_cuda_oom(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        suggestions = [
            "reduce --batch-size",
            "reduce patch/crop size in the trainer or preprocessing",
            "enable --fp16 if your model/path supports it",
            "use a smaller model or fewer workers if host memory pressure contributes",
        ]
        message = (
            "CUDA out of memory during training.\n\n"
            f"Device: {self.device}\n"
            f"Epoch: {self.current_epoch + 1}/{self.num_epochs}\n"
            "Suggested fixes:\n"
            + "\n".join(f"- {suggestion}" for suggestion in suggestions)
        )
        raise click.ClickException(message)

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
                                  "num_workers": self.num_workers, "fp16": self.fp16, "seed": self.seed,
                                  "custom_trainer_kwargs": dict(self.custom_trainer_kwargs),
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

    def _wandb_run_id_path(self) -> Path:
        return self.output_dir / "wandb_run_id.txt"

    def _load_saved_wandb_run_id(self) -> str | None:
        run_id_path = self._wandb_run_id_path()
        if not run_id_path.is_file():
            return None

        run_id = run_id_path.read_text().strip()
        return run_id or None

    def _persist_wandb_run_id(self) -> None:
        if self.wandb_run is None:
            return
        if getattr(self.wandb_run, "id", None) is None:
            return

        self._wandb_run_id_path().write_text(f"{self.wandb_run.id}\n")

    def _current_lr(self) -> float | None:
        if self.optimizer is None or not self.optimizer.param_groups:
            return None
        return float(self.optimizer.param_groups[0]["lr"])

    def _print_launch_summary(self) -> None:
        per_rank_batch = self.batch_size if self.batch_size is not None else "?"
        global_batch = (
            self.batch_size * self.world_size
            if self.batch_size is not None
            else "?"
        )
        distribution = (
            f"ddp(world_size={self.world_size}, local_rank={self.local_rank})"
            if self.distributed
            else "single-process"
        )
        model_name = (
            self.model_config.model_class.__name__
            if self.model_config is not None
            else "unknown"
        )
        loss_name = (
            self.loss_config.loss_class.__name__
            if self.loss_config is not None
            else "none"
        )
        optimizer_name = (
            self.optimizer_config.optim_class.__name__
            if self.optimizer_config is not None
            else "none"
        )
        scheduler_name = (
            self.scheduler_config.scheduler_class.__name__
            if self.scheduler_config is not None
            else "none"
        )
        resume_text = str(self.resume) if self.resume is not None else "no"
        wandb_text = (
            f"{self.wandb_project} ({self.wandb_mode or 'online'})"
            if self.wandb_project is not None
            else "off"
        )
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("Model", f"{model_name}  [dim]({loss_name})[/dim]")
        table.add_row("Runtime", f"{distribution} on {self.device}  [dim]fp16={self.fp16}[/dim]")
        table.add_row("Schedule", f"{self.num_epochs} epochs  [dim]{optimizer_name} / {scheduler_name}[/dim]")
        table.add_row("Batch", f"per-rank {per_rank_batch}  [dim]global {global_batch}[/dim]")
        table.add_row("Workers", str(self.num_workers))
        table.add_row("Resume", resume_text)
        table.add_row("W&B", wandb_text)
        table.add_row("Run Name", self.wandb_run_name)
        table.add_row("Data", str(self.input_dir))
        table.add_row("Output", str(self.output_dir))

        CONSOLE.print(
            Panel(
                table,
                title="[bold green]Training Start[/bold green]",
                border_style="green",
                expand=False,
            )
        )

    def _is_best_checkpoint(self, val_metrics: dict[str, float]) -> bool:
        if "dice" in val_metrics:
            val_dice = float(val_metrics["dice"])
            if val_dice <= self.best_val_dice:
                return False
            self.best_val_dice = val_dice
            return True

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
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            device_id=self.device,
        )
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

    def _autocast_context(self) -> Any:
        return autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        )

    def _backward_step(self, loss: torch.Tensor) -> None:
        if self.optimizer is None:
            raise RuntimeError("optimizer must be initialized before backward/step")

        self.optimizer.zero_grad()
        if self.amp_enabled:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            return

        loss.backward()
        self.optimizer.step()

    def _load_pretrain(self) -> None:
        checkpoint = torch.load(self.pretrain, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model = self._model_for_state()
        if model is None:
            raise RuntimeError("model must be initialized before loading pretrained weights")
        model.load_state_dict(state_dict, strict=False)

    def _next_train_batch(self) -> dict[str, Any]:
        if self._train_iterator is None:
            self._train_iterator = iter(self.train_loader)
        try:
            return next(self._train_iterator)
        except StopIteration:
            self._train_iterator = iter(self.train_loader)
            return next(self._train_iterator)

    def _predict_volume(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        model = self._model_for_state()
        if model is None:
            raise RuntimeError("model must be initialized before prediction")
        with self._autocast_context():
            return model.predict(images, mask)

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
                f"{self.__class__.__name__} expects preprocessed crops to be at least "
                f"{self.patch_size} voxels along each spatial dimension, got {(height, width, depth)}"
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
            raise RuntimeError(f"{self.__class__.__name__} currently supports 4-class BraTS labels only")

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
    def _dice_from_binary(pred: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
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

    def _loss_impl(self) -> Any:
        if self.loss_fn is None:
            raise RuntimeError("loss_config must be set before computing losses")
        return self.loss_fn
