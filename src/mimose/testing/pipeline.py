from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader

from mimose.datasets import DatasetType, IMFuseDataset
from mimose.models.abstract_model import AbstractModel

MASKS: list[list[bool]] = [
    [False, False, False, True],
    [False, True, False, False],
    [False, False, True, False],
    [True, False, False, False],
    [False, True, False, True],
    [False, True, True, False],
    [True, False, True, False],
    [False, False, True, True],
    [True, False, False, True],
    [True, True, False, False],
    [True, True, True, False],
    [True, False, True, True],
    [True, True, False, True],
    [False, True, True, True],
    [True, True, True, True],
]
MODALITY_NAMES = ("t1c", "t1n", "t2f", "t2w")


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val: np.ndarray | float = 0
        self.avg: np.ndarray | float = 0
        self.sum: np.ndarray | float = 0
        self.count = 0

    def update(self, val: Any, n: int = 1) -> None:
        value = np.asarray(val)
        self.sum = value * n if self.count == 0 else self.sum + (value * n)
        self.count += n
        self.avg = self.sum / self.count
        self.val = value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def softmax_output_dice_class4(
    output: torch.Tensor,
    target: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-8
    if output.ndim != 4 or target.ndim != 4:
        raise ValueError(
            "Expected output and target to have shape [B, H, W, D], "
            f"got {tuple(output.shape)} and {tuple(target.shape)}"
        )

    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1, 2, 3)) + eps
    union1 = torch.sum(o1, dim=(1, 2, 3)) + torch.sum(t1, dim=(1, 2, 3)) + eps
    net_ncr_dice = intersect1 / union1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1, 2, 3)) + eps
    union2 = torch.sum(o2, dim=(1, 2, 3)) + torch.sum(t2, dim=(1, 2, 3)) + eps
    edema_dice = intersect2 / union2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1, 2, 3)) + eps
    union3 = torch.sum(o3, dim=(1, 2, 3)) + torch.sum(t3, dim=(1, 2, 3)) + eps
    enhancing_dice = intersect3 / union3

    o4 = o3 * 0.0 if torch.sum(o3) < 500 else o3
    intersect4 = torch.sum(2 * (o4 * t3), dim=(1, 2, 3)) + eps
    union4 = torch.sum(o4, dim=(1, 2, 3)) + torch.sum(t3, dim=(1, 2, 3)) + eps
    enhancing_dice_postpro = intersect4 / union4

    o_whole = o1 + o2 + o3
    t_whole = t1 + t2 + t3
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1, 2, 3)) + eps
    union_whole = (
        torch.sum(o_whole, dim=(1, 2, 3))
        + torch.sum(t_whole, dim=(1, 2, 3))
        + eps
    )
    dice_whole = intersect_whole / union_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1, 2, 3)) + eps
    union_core = (
        torch.sum(o_core, dim=(1, 2, 3))
        + torch.sum(t_core, dim=(1, 2, 3))
        + eps
    )
    dice_core = intersect_core / union_core

    dice_separate = torch.cat(
        (
            torch.unsqueeze(net_ncr_dice, 1),
            torch.unsqueeze(edema_dice, 1),
            torch.unsqueeze(enhancing_dice, 1),
        ),
        dim=1,
    )
    dice_evaluate = torch.cat(
        (
            torch.unsqueeze(dice_whole, 1),
            torch.unsqueeze(dice_core, 1),
            torch.unsqueeze(enhancing_dice, 1),
            torch.unsqueeze(enhancing_dice_postpro, 1),
        ),
        dim=1,
    )
    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()


def run_testing(
    *,
    data_dir: Path,
    output_path: Path,
    checkpoint_path: Path,
    dataset_type: DatasetType,
    model_class: type[AbstractModel],
    model_kwargs: dict[str, Any] | None = None,
    split_file: Path,
    num_workers: int = 8,
    seed: int = 42,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Testing currently requires at least one CUDA device")
    if output_path.exists() and output_path.is_dir():
        raise ValueError(f"{output_path} must be a file path, not a directory")
    if not checkpoint_path.is_file():
        raise click.ClickException(f"Checkpoint not found: {checkpoint_path}")

    set_seed(seed)
    device = torch.device("cuda")

    try:
        split = _load_test_split(split_file=split_file, dataset_type=dataset_type)
        test_set = IMFuseDataset(
            root=data_dir,
            masking_mode=None,
            split=split,
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        if test_loader.batch_size != 1:
            raise RuntimeError("Unified test expects batch_size=1")

        model = model_class(**(model_kwargs or {}))
        if not isinstance(model, AbstractModel):
            raise RuntimeError(f"{model_class.__name__} must inherit from AbstractModel")
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)
        model.eval()
        if hasattr(model, "is_training"):
            model.is_training = False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

        total_score = AverageMeter()
        total_steps = len(MASKS) * len(test_loader)
        with torch.no_grad():
            with Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[metrics]}", style="magenta"),
                transient=True,
            ) as progress:
                task_id = progress.add_task(
                    "Testing masks",
                    total=total_steps,
                    metrics="",
                )
                for mask in MASKS:
                    mask_specific_score = AverageMeter()
                    mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

                    for batch in test_loader:
                        images = batch["images"].to(device, non_blocking=True)
                        target = batch["seg"].to(device, non_blocking=True).squeeze(1).long()
                        output = model.predict(images, mask_tensor)
                        prediction = torch.argmax(output, dim=1)
                        _, brats_dice = softmax_output_dice_class4(
                            output=prediction,
                            target=target,
                        )
                        mask_specific_score.update(brats_dice)
                        current_avg = np.asarray(mask_specific_score.avg)[0]
                        progress.update(
                            task_id,
                            advance=1,
                            metrics=(
                                f"{_mask_name(mask)}  "
                                f"WT {current_avg[0]:.4f}  "
                                f"TC {current_avg[1]:.4f}  "
                                f"ET {current_avg[2]:.4f}"
                            ),
                        )

                    mask_score_avg = np.asarray(mask_specific_score.avg)[0]
                    total_score.update(mask_score_avg)
                    _append_report_line(
                        output_path,
                        (
                            f"Available modals = {_mask_name(mask):<21}--> "
                            f"WT = {mask_score_avg[0]:.4f}, "
                            f"TC = {mask_score_avg[1]:.4f}, "
                            f"ET = {mask_score_avg[2]:.4f}, "
                            f"ETpp = {mask_score_avg[3]:.4f}"
                        ),
                    )

        avg_total_score = np.asarray(total_score.avg)
        _append_report_line(
            output_path,
            (
                f"Avg scores {'':<29}--> "
                f"WT = {avg_total_score[0]:.4f}, "
                f"TC = {avg_total_score[1]:.4f}, "
                f"ET = {avg_total_score[2]:.4f}, "
                f"ETpp = {avg_total_score[3]:.4f}"
            ),
        )
        _write_excel_summary(output_path)
        return output_path
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from None


def _load_test_split(*, split_file: Path, dataset_type: DatasetType) -> list[dict[str, Any]]:
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    split_payload = json.loads(split_file.read_text())
    dataset_key = str(dataset_type).lower()
    if dataset_key not in split_payload:
        raise RuntimeError(f"Dataset split '{dataset_key}' not found in {split_file}")
    dataset_splits = split_payload[dataset_key]
    return list(dataset_splits.get("test", []))


def _mask_name(mask: list[bool]) -> str:
    present = [name for name, enabled in zip(MODALITY_NAMES, mask) if enabled]
    return "_".join(present)


def _append_report_line(output_path: Path, line: str) -> None:
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")


def _write_excel_summary(results_path: Path) -> Path:
    excel_path = results_path.with_suffix(".xlsx")
    scores = {
        "ET": [],
        "TC": [],
        "WT": [],
        "order": [],
    }

    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "Avg" in line:
                parsed = _parse_avg_line(line)
                scores["order"].append(15)
                scores["ET"].append(parsed["ET"])
                scores["WT"].append(parsed["WT"])
                scores["TC"].append(parsed["TC"])
            else:
                parsed = _parse_result_line(line)
                scores["order"].append(_string_to_order(parsed["modals"]))
                scores["ET"].append(parsed["ET"] * 100)
                scores["WT"].append(parsed["WT"] * 100)
                scores["TC"].append(parsed["TC"] * 100)

    scores_sorted = {"ET": [], "TC": [], "WT": []}
    for key in scores_sorted:
        scores_sorted[key] = [value for _, value in sorted(zip(scores["order"], scores[key]))]

    for key, values in scores_sorted.items():
        values[-1] = sum(values[:-1]) / len(values[:-1])

    pd.DataFrame(scores_sorted).to_excel(excel_path, index=False)
    return excel_path


def _parse_result_line(line: str) -> dict[str, float | str]:
    left, right = line.split("-->")
    modals = left.split("=")[1].strip()

    metrics: dict[str, float | str] = {"modals": modals}
    for item in right.split(","):
        key, value = item.split("=")
        metrics[key.strip()] = float(value.strip())
    return metrics


def _parse_avg_line(line: str) -> dict[str, float]:
    _, right = line.split("-->")
    metrics: dict[str, float] = {}
    for item in right.split(","):
        key, value = item.split("=")
        metrics[key.strip()] = float(value.strip())
    return metrics


def _string_to_code(modals: str) -> int:
    code = 0
    if "t1c" in modals:
        code += 1
    if "t1n" in modals:
        code += 2
    if "t2f" in modals:
        code += 4
    if "t2w" in modals:
        code += 8
    return code


def _string_to_order(modals: str) -> int:
    code = _string_to_code(modals)
    match code:
        case 1:
            return 2
        case 2:
            return 1
        case 3:
            return 7
        case 4:
            return 0
        case 5:
            return 5
        case 6:
            return 4
        case 7:
            return 10
        case 8:
            return 3
        case 9:
            return 9
        case 10:
            return 8
        case 11:
            return 13
        case 12:
            return 6
        case 13:
            return 12
        case 14:
            return 11
        case 15:
            return 14
        case _:
            raise RuntimeError(f"invalid code passed: {modals}, {code}")


__all__ = [
    "AverageMeter",
    "MASKS",
    "MODALITY_NAMES",
    "run_testing",
    "set_seed",
    "softmax_output_dice_class4",
]
