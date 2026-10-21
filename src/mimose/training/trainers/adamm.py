from __future__ import annotations

from typing import Any

from mimose.training.trainers.mstkdnet import MSTKDTrainer


class AdaMMTrainer(MSTKDTrainer):
    """AdaMM shares MST-KDNet's teacher/student training contract (same data,
    masking, validation, and sliding-window inference); only the set of loss
    terms logged per step differs."""

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

    def _train_step(self, batch: dict[str, Any], epoch: int) -> dict[str, float]:
        images = batch["images"].to(self.device, non_blocking=True)
        seg = batch["seg"].to(self.device, non_blocking=True).long()
        mask = batch["mask"].to(self.device, non_blocking=True).bool()
        with self._autocast_context():
            target = self._seg_to_one_hot(seg)
            outputs = self.model(images, mask)
            metrics = self._loss_impl().training_loss(outputs, target, epoch)
            loss = metrics["loss"]

        self._backward_step(loss)
        return {key: float(value.item()) for key, value in metrics.items()}
