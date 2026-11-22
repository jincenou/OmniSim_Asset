#!/usr/bin/env python3
"""Copy the source snapshot into a separate workspace with explicit path mappings.

The default is a read-only plan. No model, dataset, asset, or training job is
created/downloaded. Existing destination files are never overwritten.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

REPOSITORY = Path(__file__).resolve().parents[1]


def validate_path(value: str) -> str:
    path = Path(value).expanduser().resolve()
    text = str(path)
    # Historical shell scripts contain unquoted paths. Keep remapped paths
    # safe for both Python literals and those scripts instead of guessing.
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", text):
        raise ValueError("Use absolute paths containing only ASCII letters, digits, /, _, ., and -")
    return text.rstrip("/")


def remap(text: str, mappings: dict[str, str]) -> str:
    keys = sorted(mappings, key=len, reverse=True)
    pattern = re.compile("(?:" + "|".join(re.escape(k) for k in keys) + r")(?=$|[^A-Za-z0-9_.-])")
    # One substitution prevents a mapped destination from being mapped again.
    return pattern.sub(lambda match: mappings[match.group()], text)


def make_plan(source: Path, workspace: Path, mappings: dict[str, str]) -> list[dict]:
    plan = []
    for origin in sorted(source.rglob("*")):
        if origin.is_symlink():
            raise ValueError(f"Unexpected source symlink: {origin}")
        if not origin.is_file():
            continue
        relative = origin.relative_to(source)
        target = workspace / relative
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {target}")
        # Reject existing symlink parents, including links to an asset library.
        for parent in target.parents:
            if parent == workspace.parent:
                break
            if parent.is_symlink():
                raise ValueError(f"Destination parent is a symlink: {parent}")
        raw = origin.read_text(encoding="utf-8")
        rendered = remap(raw, mappings)
        if origin.suffix == ".py":
            ast.parse(rendered, filename=str(relative))
        elif origin.suffix == ".json":
            json.loads(rendered)
        plan.append({"source": origin, "relative": relative, "target": target,
                     "content": rendered, "changed": raw != rendered,
                     "sha256": hashlib.sha256(rendered.encode()).hexdigest()})
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--envs-root", required=True, help="Directory containing named conda environments")
    parser.add_argument("--datasets-root", required=True, help="Replacement for the original /mnt/public_data")
    parser.add_argument("--path-map", type=Path, help="Optional JSON object with additional absolute prefix mappings")
    parser.add_argument("--apply", action="store_true", help="Write new source files after a successful full preflight")
    args = parser.parse_args()
    workspace = Path(validate_path(args.workspace_root))
    if workspace == REPOSITORY or REPOSITORY in workspace.parents or workspace in REPOSITORY.parents:
        parser.error("The execution workspace must be separate from the release repository")
    mappings = {
        "/mnt/ojc/miniconda3/envs": validate_path(args.envs_root),
        "/mnt/public_data": validate_path(args.datasets_root),
        "/mnt/ojc": str(workspace),
    }
    if args.path_map:
        extra = json.loads(args.path_map.read_text())
        if not isinstance(extra, dict):
            parser.error("--path-map must contain a JSON object")
        for old, new in extra.items():
            if not isinstance(old, str) or not old.startswith("/") or not isinstance(new, str):
                parser.error("Path mappings must map absolute string prefixes to absolute paths")
            mappings[old.rstrip("/")] = validate_path(new)
    plan = make_plan(REPOSITORY / "source", workspace, mappings)
    receipt = workspace / "omnisim_source_materialization.json"
    if receipt.exists():
        parser.error(f"Refusing to overwrite an earlier receipt: {receipt}")
    result = {"mode": "apply" if args.apply else "plan", "files": len(plan),
              "remapped_files": sum(row["changed"] for row in plan),
              "workspace_root": str(workspace), "mappings": mappings,
              "note": "Only source files; dependency code, data, assets, and runtime state are external."}
    if args.apply:
        for row in plan:
            row["target"].parent.mkdir(parents=True, exist_ok=True)
            with row["target"].open("x", encoding="utf-8") as stream:
                stream.write(row["content"])
            shutil.copymode(row["source"], row["target"])
        result["source_files"] = [{"path": str(row["relative"]), "sha256": row["sha256"],
                                   "remapped": row["changed"]} for row in plan]
        receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "source_files"}, indent=2))


if __name__ == "__main__":
    main()
