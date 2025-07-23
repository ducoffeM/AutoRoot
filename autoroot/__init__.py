from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("autoroot")
except PackageNotFoundError:
    # package is not installed
    pass
