from brainchmark.enums import TransformKind
from brainchmark.training.transforms.base_transforms import TransformManager
from brainchmark.training.transforms.imfuse import IMFuseTransformManager


def build_transform_manager(kind: TransformKind) -> TransformManager:
    if kind is TransformKind.IMFUSE:
        return IMFuseTransformManager()

    raise ValueError(f"Unsupported transform kind: {kind}")


__all__ = [
    "TransformManager",
    "IMFuseTransformManager",
    "TransformKind",
    "build_transform_manager",
]
