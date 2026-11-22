"""Construct the unchanged native33k configuration from training-view geometry."""
import argparse
import hashlib
import json
from pathlib import Path

import grid_context
import freeze_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    args = parser.parse_args()
    data, path = grid_context.manifest(args.label)
    config = freeze_config.make(args.label)
    config["saving_iterations"] = [33000]
    config["recipe_scope"] = (
        "Frozen full A native prefix; geometry-bound rule unchanged. "
        "No heldout scores used to choose horizon, bound or checkpoint."
    )
    output = grid_context.D / "configs" / f"{args.label}_33k.json"
    assert not output.exists()
    output.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    output.with_suffix(".sha256").write_text(hashlib.sha256(output.read_bytes()).hexdigest() + "\n")


if __name__ == "__main__":
    main()
