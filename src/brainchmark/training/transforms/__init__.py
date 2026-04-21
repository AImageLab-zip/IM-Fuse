from brainchmark.enums import TransformKind
from brainchmark.training.transforms.base_transforms import TransformManager
from brainchmark.training.transforms.imfuse import DCSegTransformManager, IMFuseTransformManager


def build_transform_manager(kind: TransformKind) -> TransformManager:
    if kind is TransformKind.IMFUSE:
        return IMFuseTransformManager()
    if kind is TransformKind.DCSEG:
        return DCSegTransformManager()

    raise ValueError(f"Unsupported transform kind: {kind}")


__all__ = [
    "TransformManager",
    "DCSegTransformManager",
    "IMFuseTransformManager",
    "TransformKind",
    "build_transform_manager",
]
