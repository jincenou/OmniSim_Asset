#!/usr/bin/env python3
"""Check snapshot hashes and source syntax without importing models or writing caches."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "provenance/source_manifest.json").read_text())
    errors = []
    counters = {"files": 0, "python": 0, "json": 0, "shell": 0}
    expected = {row["path"] for row in manifest["files"]}
    actual = {p.relative_to(root).as_posix() for p in (root / "source").rglob("*") if p.is_file()}
    if expected != actual:
        errors.append({"manifest_difference": sorted(expected ^ actual)})
    for row in manifest["files"]:
        path = root / row["path"]
        if not path.is_file() or path.is_symlink():
            errors.append({"path": row["path"], "error": "missing file or unexpected symlink"})
            continue
        counters["files"] += 1
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            errors.append({"path": row["path"], "error": "SHA256 mismatch"})
        try:
            text = content.decode("utf-8")
            if path.suffix == ".py":
                ast.parse(text, filename=row["path"])
                counters["python"] += 1
            elif path.suffix == ".json":
                json.loads(text)
                counters["json"] += 1
            elif path.suffix == ".sh":
                result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
                if result.returncode:
                    raise ValueError(result.stderr.strip())
                counters["shell"] += 1
        except (UnicodeError, SyntaxError, ValueError) as error:
            errors.append({"path": row["path"], "error": str(error)})
    print(json.dumps({"checked": counters, "errors": errors,
                      "scope": "CPU static validation only; no model imports, inference, or training"}, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
