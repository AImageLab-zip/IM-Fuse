from enum import StrEnum

from brainchmark.utils.cli_overrides import CONFIG_NONE


class DatasetType(StrEnum):
    BRATS18 = "brats18"
    BRATS23 = "brats23"


class CropMode(StrEnum):
    NONE = CONFIG_NONE
    CENTER = "center"
    NON_EMPTY = "non_empty"


class ClampMode(StrEnum):
    NONE = "none"
    SUBJECT = "subject"
    DATASET = "dataset"


class NormMode(StrEnum):
    NONE = "none"
    MIN_MAX = "min_max"
    SUBJECT_ZSCORE = "subject_zscore"
    DATASET_ZSCORE = "dataset_zscore"


class TrainerKind(StrEnum):
    IMFUSE = "imfuse"


class ModelKind(StrEnum):
    IMFUSE = "imfuse"
    MMFORMER = "mmformer"


class TransformKind(StrEnum):
    IMFUSE = "imfuse"


class OptimizerKind(StrEnum):
    RADAM = "radam"
    ADAMW = "adamw"
    SGD = "sgd"
    ADAM = "adam"


class SchedulerKind(StrEnum):
    POLY = "poly"
    COSINE = "cosine"
    STEP = "step"
    MULTISTEP = "multistep"
    PLATEAU = "plateau"


class LossKind(StrEnum):
    IMFUSE = "imfuse"


class MaskingMode(StrEnum):
    RANDOM = "random"
    VALIDATION = "validation"
    TEST = "test"
