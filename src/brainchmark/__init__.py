from importlib.metadata import PackageNotFoundError, version

PACKAGE_NAME = "BrainchMark"

try:
    __version__ = version(PACKAGE_NAME)
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
