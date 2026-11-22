"""Completed primary and alternative windows; package counts are not people."""
from scan_native import ROOT


def ready_paths():
    return sorted([
        *(ROOT / "samples").glob("*/*/*/READY.json"),
        *(ROOT / "retry").glob("*/*/*/*/READY.json"),
    ])
