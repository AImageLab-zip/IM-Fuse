import importlib
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from brainchmark.training.config import IMFuseTrainingConfig, OptimizerKind, SchedulerKind


LOGGER = logging.getLogger(__name__)


class IMFuseTrainer:
    def __init__(self, config: IMFuseTrainingConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model: torch.nn.Module | None = None
        self.optimizer: Optimizer | None = None
        self.scheduler: Any | None = None
        self.train_loader: Any | None = None
        self.val_loader: Any | None = None
        self.test_loader: Any | None = None
        self.wandb_run: Any | None = None
        self.best_val_dice = float("-inf")

    def fit(self) -> None:
        self._configure_logging()
        self._setup_seed()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = self._build_model()
        self.optimizer = self._build_optimizer(self.model)
        self.scheduler = self._build_scheduler(self.optimizer)
        self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders()
        self._maybe_load_checkpoint_or_pretrain()
        self._init_wandb()

        start_time = time.time()
        train_iter = iter(self.train_loader)
        start_epoch = self._load_resume_epoch()

        for epoch in range(start_epoch, self.config.num_epochs):
            step_lr = self._step_scheduler(epoch)
            epoch_metrics: dict[str, float] = {
                "fusecross": 0.0,
                "fusedice": 0.0,
                "sepcross": 0.0,
                "sepdice": 0.0,
                "prmcross": 0.0,
                "prmdice": 0.0,
                "loss": 0.0,
            }

            self.model.train()
            self._set_aux_training_flag(True)
            iter_per_epoch = self.config.iter_per_epoch or len(self.train_loader)

            for iteration in range(iter_per_epoch):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                batch_metrics = self._train_step(batch, epoch)
                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value

                LOGGER.info(
                    "Epoch %s/%s Iter %s/%s Loss %.4f fusecross:%.4f fusedice:%.4f "
                    "sepcross:%.4f sepdice:%.4f prmcross:%.4f prmdice:%.4f",
                    epoch + 1,
                    self.config.num_epochs,
                    iteration + 1,
                    iter_per_epoch,
                    batch_metrics["loss"],
                    batch_metrics["fusecross"],
                    batch_metrics["fusedice"],
                    batch_metrics["sepcross"],
                    batch_metrics["sepdice"],
                    batch_metrics["prmcross"],
                    batch_metrics["prmdice"],
                )

                if self.config.debug:
                    break

            divisor = 1 if self.config.debug else iter_per_epoch
            averaged = {key: value / divisor for key, value in epoch_metrics.items()}
            self._log_wandb_train(epoch, averaged, step_lr)
            self._save_checkpoint(epoch, best=False)

            should_validate = self.config.debug or (epoch + 1) in self.config.val_check
            if should_validate:
                self._run_validation_and_test(epoch)

            if self.config.debug:
                break

        LOGGER.info("total time: %.4f hours", (time.time() - start_time) / 3600)
        if self.wandb_run is not None:
            self.wandb_run.finish()

    def _configure_logging(self) -> None:
        if logging.getLogger().handlers:
            return
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    def _resolve_device(self, requested: str | None) -> torch.device:
        if requested:
            return torch.device(requested)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _legacy(self, module_name: str) -> Any:
        return importlib.import_module(module_name)

    def _setup_seed(self) -> None:
        random_seed = self._legacy("IMFuse.utils.random_seed")
        random_seed.setup_seed(self.config.seed)

    def _build_model(self) -> torch.nn.Module:
        if self.config.first_skip:
            model_module = self._legacy("IMFuse.IMFuse")
            model = model_module.IMFuse(
                num_cls=self.config.num_classes,
                interleaved_tokenization=self.config.interleaved_tokenization,
                mamba_skip=self.config.mamba_skip,
            )
        else:
            model_module = self._legacy("IMFuse.IMFuse_no1skip")
            model = model_module.Model(
                num_cls=self.config.num_classes,
                interleaved_tokenization=self.config.interleaved_tokenization,
                mamba_skip=self.config.mamba_skip,
            )

        if self.device.type == "cuda":
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model).cuda()
            else:
                model = model.cuda()
        else:
            model = model.to(self.device)

        return model

    def _build_optimizer(self, model: torch.nn.Module) -> Optimizer:
        train_params = [
            {
                "params": model.parameters(),
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
            }
        ]

        if self.config.optimizer is OptimizerKind.RADAM:
            return torch.optim.RAdam(train_params)
        if self.config.optimizer is OptimizerKind.ADAMW:
            return torch.optim.AdamW(train_params)
        if self.config.optimizer is OptimizerKind.SGD:
            return torch.optim.SGD(train_params, momentum=0.9)
        raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def _build_scheduler(self, optimizer: Optimizer) -> Any:
        if self.config.scheduler is SchedulerKind.COSINE:
            return CosineAnnealingLR(optimizer, T_max=self.config.num_epochs)
        scheduler_module = self._legacy("IMFuse.utils.lr_scheduler")
        return scheduler_module.LR_Scheduler(self.config.lr, self.config.num_epochs)

    def _build_dataloaders(self) -> tuple[Any, Any, Any]:
        datasets_module = self._legacy("IMFuse.data.datasets_nii")
        data_utils_module = self._legacy("IMFuse.data.data_utils")
        scheduler_module = self._legacy("IMFuse.utils.lr_scheduler")

        train_file, val_file, test_file = self.config.resolved_split_files()

        train_set = datasets_module.Brats_loadall_nii(
            transforms=self.config.train_transforms,
            root=str(self.config.input_dir),
            num_cls=self.config.num_classes,
            train_file=train_file,
        )
        val_set = datasets_module.Brats_loadall_val_nii(
            transforms=self.config.test_transforms,
            root=str(self.config.input_dir),
            num_cls=self.config.num_classes,
            val_file=val_file,
        )
        test_set = datasets_module.Brats_loadall_test_nii(
            transforms=self.config.test_transforms,
            root=str(self.config.input_dir),
            num_cls=self.config.num_classes,
            test_file=test_file,
        )

        loader_cls = scheduler_module.MultiEpochsDataLoader
        train_loader = loader_cls(
            dataset=train_set,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            shuffle=True,
            worker_init_fn=data_utils_module.init_fn,
        )
        val_loader = loader_cls(
            dataset=val_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        test_loader = loader_cls(
            dataset=test_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )
        return train_loader, val_loader, test_loader

    def _model_core(self) -> torch.nn.Module:
        assert self.model is not None
        return self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

    def _set_aux_training_flag(self, enabled: bool) -> None:
        core = self._model_core()
        if hasattr(core, "is_training"):
            core.is_training = enabled

    def _maybe_load_checkpoint_or_pretrain(self) -> None:
        if self.config.resume is not None:
            return
        if self.config.pretrain is None:
            return
        checkpoint = torch.load(self.config.pretrain, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        assert self.model is not None
        self.model.load_state_dict(state_dict, strict=False)
        LOGGER.info("Loaded pretrained weights from %s", self.config.pretrain)

    def _load_resume_epoch(self) -> int:
        if self.config.resume is None:
            return 0
        checkpoint = torch.load(self.config.resume, map_location=self.device)
        assert self.model is not None
        assert self.optimizer is not None
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optim_dict"])
        self.best_val_dice = checkpoint.get("val_Dice_best", float("-inf"))
        LOGGER.info("Resumed from %s at epoch %s", self.config.resume, checkpoint["epoch"])
        return int(checkpoint["epoch"]) + 1

    def _init_wandb(self) -> None:
        if not self.config.wandb.enabled:
            return
        import wandb

        slurm_job_id = os.getenv("SLURM_JOB_ID")
        run_name = (
            f"{self.config.dataname}_IMFuse"
            f"{'no_1_skip' if not self.config.first_skip else ''}_"
            f"{'Interleaved' if self.config.interleaved_tokenization else ''}"
            f"{'Skip' if self.config.mamba_skip else ''}_jobid{slurm_job_id}"
        )
        config_dict = asdict(self.config)
        config_dict["input_dir"] = str(self.config.input_dir)
        config_dict["output_dir"] = str(self.config.output_dir)
        if self.config.resume is not None:
            config_dict["resume"] = str(self.config.resume)
        if self.config.pretrain is not None:
            config_dict["pretrain"] = str(self.config.pretrain)

        self.wandb_run = wandb.init(
            project=self.config.wandb.project,
            name=run_name,
            id=run_name,
            mode=self.config.wandb.mode,
            resume=self.config.wandb.resume,
            config=config_dict,
        )

    def _step_scheduler(self, epoch: int) -> float:
        assert self.optimizer is not None
        if self.config.scheduler is SchedulerKind.COSINE:
            assert isinstance(self.scheduler, CosineAnnealingLR)
            self.scheduler.step(epoch)
            return float(self.optimizer.param_groups[0]["lr"])
        return float(self.scheduler(self.optimizer, epoch))

    def _train_step(self, batch: Any, epoch: int) -> dict[str, float]:
        assert self.model is not None
        assert self.optimizer is not None

        criterions = self._legacy("IMFuse.utils.criterions")
        x, target, mask = batch[:3]
        x = x.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)

        fuse_pred, sep_preds, prm_preds = self.model(x, mask)

        fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=self.config.num_classes)
        fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=self.config.num_classes)
        fuse_loss = fuse_cross_loss + fuse_dice_loss

        sep_cross_loss = torch.zeros(1, device=self.device).float()
        sep_dice_loss = torch.zeros(1, device=self.device).float()
        for sep_pred in sep_preds:
            sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=self.config.num_classes)
            sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=self.config.num_classes)
        sep_loss = sep_cross_loss + sep_dice_loss

        prm_cross_loss = torch.zeros(1, device=self.device).float()
        prm_dice_loss = torch.zeros(1, device=self.device).float()
        for prm_pred in prm_preds:
            prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=self.config.num_classes)
            prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=self.config.num_classes)
        prm_loss = prm_cross_loss + prm_dice_loss

        if epoch < self.config.region_fusion_start_epoch:
            loss = sep_loss + prm_loss
        else:
            loss = fuse_loss + sep_loss + prm_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "fusecross": float(fuse_cross_loss.item()),
            "fusedice": float(fuse_dice_loss.item()),
            "sepcross": float(sep_cross_loss.item()),
            "sepdice": float(sep_dice_loss.item()),
            "prmcross": float(prm_cross_loss.item()),
            "prmdice": float(prm_dice_loss.item()),
        }

    def _save_checkpoint(self, epoch: int, best: bool) -> None:
        assert self.model is not None
        assert self.optimizer is not None
        filename = "best.pth" if best else "model_last.pth"
        file_path = self.config.output_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optim_dict": self.optimizer.state_dict(),
                "val_Dice_best": self.best_val_dice,
            },
            file_path,
        )

    def _run_validation_and_test(self, epoch: int) -> None:
        assert self.model is not None
        assert self.val_loader is not None
        assert self.test_loader is not None
        predict_module = self._legacy("IMFuse.predict")

        LOGGER.info("validate ...")
        with torch.no_grad():
            val_scores, val_loss = predict_module.test_softmax(
                self.val_loader,
                self.model,
                dataname=self.config.dataname,
            )
        val_wt, val_tc, val_et, val_etpp = val_scores
        val_dice = (val_et + val_wt + val_tc) / 3
        self._log_wandb_eval(
            prefix="val",
            epoch=epoch,
            wt=val_wt.item(),
            tc=val_tc.item(),
            et=val_et.item(),
            etpp=val_etpp.item(),
            dice=val_dice.item(),
            seg_loss=val_loss.cpu().item(),
        )
        if val_dice.item() > self.best_val_dice:
            self.best_val_dice = val_dice.item()
            LOGGER.info("save best model ...")
            self._save_checkpoint(epoch, best=True)

        LOGGER.info("testing ...")
        with torch.no_grad():
            test_scores, test_loss = predict_module.test_softmax(
                self.test_loader,
                self.model,
                dataname=self.config.dataname,
            )
        test_wt, test_tc, test_et, test_etpp = test_scores
        test_dice = (test_et + test_wt + test_tc) / 3
        self._log_wandb_eval(
            prefix="test",
            epoch=epoch,
            wt=test_wt.item(),
            tc=test_tc.item(),
            et=test_et.item(),
            etpp=test_etpp.item(),
            dice=test_dice.item(),
            seg_loss=test_loss.cpu().item(),
        )

        self.model.train()
        self._set_aux_training_flag(True)

    def _log_wandb_train(self, epoch: int, metrics: dict[str, float], step_lr: float) -> None:
        if self.wandb_run is None:
            return
        self.wandb_run.log(
            {
                "train/epoch": epoch,
                "train/loss": metrics["loss"],
                "train/fusecross": metrics["fusecross"],
                "train/fusedice": metrics["fusedice"],
                "train/sepcross": metrics["sepcross"],
                "train/sepdice": metrics["sepdice"],
                "train/prmcross": metrics["prmcross"],
                "train/prmdice": metrics["prmdice"],
                "train/learning_rate": step_lr,
            }
        )

    def _log_wandb_eval(
        self,
        prefix: str,
        epoch: int,
        wt: float,
        tc: float,
        et: float,
        etpp: float,
        dice: float,
        seg_loss: float,
    ) -> None:
        LOGGER.info(
            "%s epoch = %s, WT = %.2f, TC = %.2f, ET = %.2f, ETpp = %.2f, loss = %.2f",
            prefix.capitalize(),
            epoch,
            wt,
            tc,
            et,
            etpp,
            seg_loss,
        )
        if self.wandb_run is None:
            return
        self.wandb_run.log(
            {
                f"{prefix}/epoch": epoch,
                f"{prefix}/{prefix}_WT_Dice": wt,
                f"{prefix}/{prefix}_TC_Dice": tc,
                f"{prefix}/{prefix}_ET_Dice": et,
                f"{prefix}/{prefix}_ETpp_Dice": etpp,
                f"{prefix}/{prefix}_Dice": dice,
                f"{prefix}/seg_loss": seg_loss,
            }
        )
