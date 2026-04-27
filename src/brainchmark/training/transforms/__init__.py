from brainchmark.enums import TransformKind
from brainchmark.training.transforms.base_transforms import TransformManager
from brainchmark.training.transforms.imfuse import (
    DCSegTransformManager,
    IMFuseTransformManager,
    RFNetTransformManager,
)


def build_transform_manager(kind: TransformKind) -> TransformManager:
    if kind is TransformKind.IMFUSE:
        return IMFuseTransformManager()
    if kind is TransformKind.DCSEG:
        return DCSegTransformManager()
    if kind is TransformKind.RFNET:
        return RFNetTransformManager()

    raise ValueError(f"Unsupported transform kind: {kind}")


__all__ = [
    "TransformManager",
    "DCSegTransformManager",
    "IMFuseTransformManager",
    "RFNetTransformManager",
    "TransformKind",
    "build_transform_manager",
]
