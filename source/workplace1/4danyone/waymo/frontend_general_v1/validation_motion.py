"""Frozen independent-trajectory motion evaluation; no output-based recipe choice."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

G = Path(__file__).resolve().parent
V = G / "validation12"


def frozen(suite, name="PROTOCOL"):
    assert suite == "real"
    path = V / ("MOTION_PROTOCOL.json" if name == "PROTOCOL" else "MOTION_RETARGET_PROTOCOL.json")
    assert name in ("PROTOCOL", "RETARGET_PROTOCOL")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == path.with_suffix(".sha256").read_text().strip()
    return path, json.loads(path.read_text())


def paths(suite, case):
    _, cfg = frozen(suite)
    assert case in cfg["cases"]
    return Path(cfg["sources"][case]), V / "motion/cases" / case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample")
    parser.add_argument("stage", choices=["prepare", "sample", "retarget"])
    args = parser.parse_args()
    _, cfg = frozen("real")
    assert args.sample in cfg["cases"]
    for filename, digest in cfg["code_and_source_sha256"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename
    name = {"prepare": "prepare_condmdi_case", "sample": "sample_condmdi_case", "retarget": "retarget_condmdi"}[args.stage]
    module = importlib.import_module(name)
    module.frozen_protocol = frozen
    module.case_paths = paths
    sys.argv = [name, args.sample]
    if args.stage != "prepare":
        sys.argv += ["text"]
    sys.argv += ["--suite", "real"]
    module.main()
    if args.stage == "prepare":
        from audit_condmdi_retarget import audit
        audit(paths("real", args.sample)[1])


if __name__ == "__main__":
    main()
