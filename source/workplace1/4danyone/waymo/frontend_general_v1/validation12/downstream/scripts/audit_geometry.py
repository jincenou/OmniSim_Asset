"""Check source-to-normalized projection equivalence before any training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parents[1]


def project(points, matrix, K):
    p = np.c_[points, np.ones(len(points))] @ np.linalg.inv(matrix).T
    q = p[:, :3] @ K.T
    assert np.all(q[:, 2] > 0)
    return q[:, :2] / q[:, 2:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    args = parser.parse_args()
    root = D / "four_dg/data" / args.label
    data = json.loads((root / "fdanyone.json").read_text())
    source = json.loads((D / "sources" / args.label / "cameras.json").read_text())
    init = np.load(D / "data" / args.label / "init.npz")
    points = init["positions"][np.isclose(init["times"], 0)]
    normalized = np.load(root / "canonical.npz")["points"]
    T = np.asarray(data["source_to_normalized"])
    transformed = (np.c_[points, np.ones(len(points))] @ T.T)[:, :3]
    assert np.max(np.abs(transformed - normalized)) < 2e-6
    errors = []
    for original, target in zip(source["cameras"], data["cameras"], strict=True):
        K = np.array(original["K"])
        K[:2] *= 0.5
        a = project(points, np.array(original["camera_to_world"]), K)
        b = project(normalized, np.array(target["camtoworld"]), np.array(target["K"]))
        error = float(np.abs(a - b).max())
        assert error <= 1e-3, (args.label, target["view"], error)
        errors.append(error)
    (root / "geometry_audit.json").write_text(json.dumps({
        "state": "all24_camera_normalization_projection_passed",
        "points_checked_per_view": len(points), "max_abs_pixels": max(errors),
        "threshold_pixels": 1e-3, "quality_approved": False,
        "scope": "Numerical projection equivalence, not real-world pose or scale accuracy.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
