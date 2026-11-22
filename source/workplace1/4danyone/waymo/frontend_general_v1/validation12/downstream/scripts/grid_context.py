"""Bind the existing fixed Grid4D implementation to this cohort's data only."""
import json
from pathlib import Path
import sys

D = Path(__file__).resolve().parents[1]
GRID = D.parents[2] / "grid4d"
sys.path.insert(0, str(GRID / "scripts"))
import grid_common


def manifest(label):
    spec = json.loads((D / "PREP_PROTOCOL.json").read_text())
    assert label in {f"{s}_{a}" for s in spec["subjects"] for a in ("G0", "G2")}
    path = D / "four_dg/data" / label / "fdanyone.json"
    data = json.loads(path.read_text())
    assert data["heldout_views"] == [0, 8, 16]
    assert data["train_views"] == [v for v in range(24) if v not in (0, 8, 16)]
    assert data["time_denominator"] == data["frame_count"] - 1
    assert (data["width"], data["height"]) == (352, 640)
    assert Path(data["source_data"]).resolve() == D / "data" / label
    return data, path


grid_common.manifest = manifest
grid_common.BASE = D / "data"
# Select the local storage/guard adapter; native model/source stay at the pinned GRID source.
sys.path.insert(0, str(D / "scripts"))
