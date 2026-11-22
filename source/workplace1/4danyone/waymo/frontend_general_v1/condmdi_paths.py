"""Explicit frozen experiment namespaces; never infer an input from output."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def suite_root(suite):
    if suite not in ["counterfactual", "real"]:
        raise ValueError(suite)
    return ROOT / "text_motion" / "real" if suite == "real" else ROOT / "text_motion"


def frozen_protocol(suite, name="PROTOCOL"):
    path = suite_root(suite) / (name + ".json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == path.with_suffix(".sha256").read_text().strip()
    return path, json.loads(path.read_text())


def case_paths(suite, case):
    _, protocol = frozen_protocol(suite)
    assert case in protocol["cases"]
    if suite == "real":
        source = Path(protocol["sources"][case])
        assert source.is_relative_to(ROOT.parent / "scene20/r20261019/scenes")
    else:
        source = ROOT / "occlusion_v1" / case / "rgb"
    return source, suite_root(suite) / "cases" / case
