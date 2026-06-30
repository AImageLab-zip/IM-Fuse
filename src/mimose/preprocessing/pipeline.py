# Standard library
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import click

# External dependencies
import medpy.io as medio
from medpy.core import ImageLoadingError
import numpy as np
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
from rich.prompt import Confirm
from rich.table import Table

# Internal modules
from mimose.preprocessing.config import CropConfig, ClampConfig, NormConfig

CONSOLE = Console()

def preprocess_case(
    file: dict[str, Path | str],
    output_dir: Path,
    crop_config: CropConfig,
    clamp_config: ClampConfig,
    norm_config: NormConfig,
) -> str:
    output_file = output_dir / f"{file['name']}.npz"

    modals = ["t1c", "t1n", "t2f", "t2w"]

    image_files = []

    for modal in modals:
        image, _ = medio.load(file[modal])  # type: ignore[index]
        image_files.append(image)
    images = np.stack(image_files, axis=0)

    seg, _ = medio.load(file["seg"]) # type: ignore[index]
    seg = np.expand_dims(seg,axis=0)

    images, seg = crop_config.fn(images,seg,crop_config)
    images = clamp_config.fn(images,clamp_config)
    images = norm_config.fn(images,norm_config)
    np.savez_compressed(output_file, images=images.astype(np.float32), seg=seg.astype(np.uint8))
    return str(output_file)

def run_preprocessing(
    input_dir: Path,
    output_dir: Path,
    crop_config: CropConfig,
    clamp_config: ClampConfig,
    norm_config: NormConfig,
    yes: bool
) -> None:
    """Run the preprocessing pipeline with the selected crop configuration."""
    if output_dir.exists():
        if not output_dir.is_dir():
            raise click.BadParameter(
                f"'{output_dir}' is not a valid directory.",
                param_hint="output_dir",
            )

        if not yes:
            table = Table.grid(padding=(0, 2))
            table.add_column(style="bold yellow", no_wrap=True)
            table.add_column(style="white")
            table.add_row("Action", "Delete existing preprocessing output")
            table.add_row("Target", str(output_dir))
            table.add_row("Effect", "This folder will be removed before preprocessing starts")
            CONSOLE.print(
                Panel(
                    table,
                    title="[bold yellow]Confirmation Required[/bold yellow]",
                    border_style="yellow",
                    expand=False,
                )
            )
            confirmed = Confirm.ask(
                "[bold yellow]Delete the existing output directory?[/bold yellow]",
                default=False,
                console=CONSOLE,
            )
            if not confirmed:
                raise click.Abort()

        shutil.rmtree(output_dir)
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("Deleted", str(output_dir))
        CONSOLE.print(
            Panel(
                table,
                title="[bold green]Preprocessing Output Reset[/bold green]",
                border_style="green",
                expand=False,
            )
        )

    output_dir.mkdir(parents=True)

    # Getting the file list:
    input_files = []
    try:
        for sub in input_dir.iterdir():
            if sub.is_dir():
                input_files.append({
                    'name':sub.name,
                    't1c':sub/f'{sub.name}-t1c.nii.gz',
                    't1n':sub/f'{sub.name}-t1n.nii.gz',
                    't2f':sub/f'{sub.name}-t2f.nii.gz',
                    't2w':sub/f'{sub.name}-t2w.nii.gz',
                    'seg':sub/f'{sub.name}-seg.nii.gz'
                })


        num_workers = len(os.sched_getaffinity(0))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(preprocess_case, file, output_dir, crop_config,clamp_config,norm_config)
                for file in input_files
            ]

            with Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
            ) as progress:
                task_id = progress.add_task(
                    f"Preprocess {len(input_files)} cases",
                    total=len(futures),
                )
                for future in as_completed(futures):
                    future.result()
                    progress.update(task_id, advance=1)

        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("Cases", str(len(input_files)))
        table.add_row("Output", str(output_dir))
        CONSOLE.print(
            Panel(
                table,
                title="[bold green]Preprocessing Complete[/bold green]",
                border_style="green",
                expand=False,
            )
        )
    except FileNotFoundError as e:
        raise click.FileError(e.filename, hint="File not found")
    except ImageLoadingError as e:
        raise click.ClickException(str(e))

    
    

                
