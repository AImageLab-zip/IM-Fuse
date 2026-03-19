# Standard library
from pathlib import Path

# External dependencies
from rich.console import Console
from rich.text import Text
import typer

# Internal modules
from brainchmark.preprocessing import (
    CropMode,
    build_crop_config,
    run_preprocessing,
    DatasetType
)

app = typer.Typer(help="BrainchMark CLI")
console = Console()


@app.callback()
def main() -> None:
    """BrainchMark command group."""


@app.command()
def hello() -> None:
    """Simple test command."""
    message = "Hello from the AImageLab Team!"

    text = Text(message, style="bold magenta")
    text.stylize("bold cyan", 0, 5)      # "Hello"
    text.stylize("bold yellow", 15, 24)  # "AImageLab"

    console.print()
    console.print(text, justify="center")
    console.print()


@app.command()
def preprocess(
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        file_okay=False,
        dir_okay=True,
        exists=True,
        readable=True,
        help="Directory containing the input data to preprocess.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        help="Directory where preprocessed data will be written.",
    ),
    dataset_type: str = typer.Option(
        ...,
        "--dataset-type",
        help="Input dataset type. Choose either brats18 or brats23",
    ),
    crop_mode: CropMode = typer.Option(
        CropMode.NONE,
        "--crop-mode",
        help="Cropping strategy: none, center, or non-empty.",
    ),
    crop_size: tuple[int, int, int] | None = typer.Option(
        None,
        "--crop-size",
        help="Center crop size as three integers: X Y Z.",
    ),
    crop_min_size: tuple[int, int, int] | None = typer.Option(
        None,
        "--crop-min-size",
        help="Minimum non-empty crop size as three integers: X Y Z.",
    ),
) -> None:
    """Run dataset preprocessing."""
    crop_config = build_crop_config(
        crop_mode=crop_mode,
        crop_size=crop_size,
        crop_min_size=crop_min_size,
    )

    typer.echo(
        "Preprocessing data "
        f"from {input_dir} to {output_dir} "
        f"assuming a '{dataset_type}' configuration "
        f"with crop mode '{crop_config.mode}'."
    )
    run_preprocessing(
        input_dir=input_dir,
        output_dir=output_dir,
        dataset_type=dataset_type,
        crop_config=crop_config,
    )
