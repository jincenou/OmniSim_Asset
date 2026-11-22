"""Output retention only; no changes to training or metric computations."""
import shutil
from pathlib import Path


class NullVideoWriter:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def append_data(self, value):
        pass


def discard_sam2_input(destination):
    """Call only after all mask workers have exited successfully."""
    directory = Path(destination) / "sam2_input"
    assert not directory.is_symlink()
    if directory.exists():
        shutil.rmtree(directory)
