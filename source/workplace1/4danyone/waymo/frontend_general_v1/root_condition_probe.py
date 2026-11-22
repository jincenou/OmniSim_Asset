"""Isolated box-assisted root-condition diagnostic, using frozen existing stages."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

G = Path(__file__).resolve().parent
E = G / "text_motion/bilateral_context/root_condition_v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "sample", "retarget", "align"])
    parser.add_argument("--branch", choices=["no_text", "text"])
    args = parser.parse_args()
    protocol = E / "PROTOCOL.json"
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == protocol.with_suffix(".sha256").read_text().strip()
    cfg = json.loads(protocol.read_text())
    for filename, checksum in cfg["code_and_source_sha256"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == checksum, filename
    if args.stage == "align":
        import align_bilateral_boxes as module
        module.E = E
        sys.argv = ["align_bilateral_boxes", args.branch, "box_assisted"]
    else:
        import bilateral_condmdi as module
        module.E = E
        sys.argv = ["bilateral_condmdi", args.stage]
        if args.branch:
            sys.argv += ["--branch", args.branch]
    module.main()


if __name__ == "__main__":
    main()
