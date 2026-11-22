"""Explicit experiment-path adapter for the already verified CondMDI functions."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

G = Path(__file__).resolve().parent
E = G / "text_motion/bilateral_context/repair_v1"


def frozen(suite, name="PROTOCOL"):
    if suite != "real":
        raise ValueError("Only the registered real-context pilot")
    path = E / (name + ".json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == path.with_suffix(".sha256").read_text().strip()
    cfg = json.loads(path.read_text())
    return path, cfg


def paths(suite, case):
    _, cfg = frozen(suite)
    assert case in cfg["cases"] and suite == "real"
    return E / "source", E / "cases" / case


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "sample", "retarget"])
    p.add_argument("--branch", choices=["no_text", "text"])
    args = p.parse_args()
    _, cfg = frozen("real")
    for name, digest in cfg["code_and_source_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    names = {"prepare": "prepare_condmdi_case", "sample": "sample_condmdi_case", "retarget": "retarget_condmdi"}
    module = importlib.import_module(names[args.stage])
    # Replace only experiment lookup, not model math, masks, normalization or RNG.
    module.frozen_protocol = frozen
    module.case_paths = paths
    sys.argv = [names[args.stage], "P18", "--suite", "real"]
    if args.stage != "prepare":
        if args.branch is None:
            raise ValueError("Missing explicit branch")
        sys.argv.insert(2, args.branch)
    module.main()
    if args.stage == "prepare":
        from audit_condmdi_retarget import audit
        audit(E / "cases/P18")


if __name__ == "__main__":
    main()
