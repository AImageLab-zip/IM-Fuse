from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


DEFAULT_TRAIN_TRANSFORMS = (
    "Compose([RandCrop3D((128,128,128)), RandomRotion(10), "
    "RandomIntensityChange((0.1,0.1)), RandomFlip(0), "
    "NumpyType((np.float32, np.int64)),])"
)
DEFAULT_TEST_TRANSFORMS = "Compose([NumpyType((np.float32, np.int64)),])"


class TrainerKind(StrEnum):
    IMFUSE = "imfuse"


class OptimizerKind(StrEnum):
    RADAM = "radam"
    ADAMW = "adamw"
    SGD = "sgd"


class SchedulerKind(StrEnum):
    POLY = "poly"
    COSINE = "cosine"


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = True
    project: str = "SegmentationMM"
    mode: str = "online"
    resume: str = "allow"


@dataclass(frozen=True)
class IMFuseTrainingConfig:
    input_dir: Path
    output_dir: Path
    trainer: TrainerKind = TrainerKind.IMFUSE
    dataname: str = "BRATS2018"
    batch_size: int = 1
    lr: float = 2e-4
    weight_decay: float = 3e-5
    optimizer: OptimizerKind = OptimizerKind.RADAM
    scheduler: SchedulerKind = SchedulerKind.POLY
    num_epochs: int = 1000
    num_workers: int = 8
    iter_per_epoch: int | None = None
    region_fusion_start_epoch: int = 0
    seed: int = 999
    resume: Path | None = None
    pretrain: Path | None = None
    debug: bool = False
    interleaved_tokenization: bool = False
    mamba_skip: bool = False
    first_skip: bool = False
    device: str | None = None
    train_transforms: str = DEFAULT_TRAIN_TRANSFORMS
    test_transforms: str = DEFAULT_TEST_TRANSFORMS
    train_file: str | None = None
    val_file: str | None = None
    test_file: str | None = None
    val_check: tuple[int, ...] = field(
        default_factory=lambda: (
            50,
            100,
            150,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            850,
            900,
            910,
            920,
            930,
            940,
            950,
            955,
            960,
            965,
            970,
            975,
            980,
            985,
            990,
            995,
            1000,
        )
    )
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.iter_per_epoch is not None and self.iter_per_epoch <= 0:
            raise ValueError("iter_per_epoch must be > 0 when provided")
        if self.region_fusion_start_epoch < 0:
            raise ValueError("region_fusion_start_epoch must be >= 0")

    @property
    def num_classes(self) -> int:
        if self.dataname in {"BRATS2023", "BRATS2021", "BRATS2020", "BRATS2018"}:
            return 4
        if self.dataname == "BRATS2015":
            return 5
        raise ValueError(f"Unsupported dataname: {self.dataname}")

    def resolved_split_files(self) -> tuple[str, str, str]:
        if self.train_file and self.val_file and self.test_file:
            return self.train_file, self.val_file, self.test_file

        if self.dataname in {"BRATS2023", "BRATS2020", "BRATS2015"}:
            return (
                self.train_file or "datalist/train.txt",
                self.val_file or "datalist/val15splits.csv",
                self.test_file or "datalist/test15splits2.csv",
            )
        if self.dataname == "BRATS2018":
            return (
                self.train_file or "datalist/train3.txt",
                self.val_file or "datalist/Brats18_val15splits.csv",
                self.test_file or "datalist/Brats18_test15splits.csv",
            )
        raise ValueError(f"Unsupported dataname: {self.dataname}")
