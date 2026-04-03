from enum import StrEnum

class DatasetType(StrEnum):
    BRATS18 = "brats18"
    BRATS23 = "brats23"

class MaskingMode(StrEnum):
    RANDOM = "random"
    VALIDATION = "validation"
    TEST = "test"