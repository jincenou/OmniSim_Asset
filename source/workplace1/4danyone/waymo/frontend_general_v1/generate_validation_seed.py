"""Explicit generation seed adapter; existing stage instrumentation is unchanged."""
import json
import os
from pathlib import Path
import sys

G = Path(__file__).resolve().parent
sys.path.insert(0, os.environ["FDANYONE_RUNTIME"])


def main():
    import inference
    import generate_dev6
    cfg = json.loads((G / "validation12/second_seed/PROTOCOL.json").read_text())
    root = Path(os.environ["FDANYONE_WAYMO_ROOT"])
    assert root.parent.name in cfg["subjects"] and root.name in ("G0", "G2")
    seed = cfg["seed"]
    assert seed == 314159
    original = inference.inference

    def seeded(*args, **kwargs):
        assert kwargs["seed"] == 42, "Underlying adapter changed; review required"
        kwargs["seed"] = seed
        return original(*args, **kwargs)

    inference.inference = seeded
    generate_dev6.main()
    metadata = json.loads((root / "outputs/turbo/metadata.json").read_text())
    assert metadata["generation"]["seed"] == seed


if __name__ == "__main__":
    main()
