"""Verify the geometry needed to return standardized proposals to the subject.

Uses only the pre-sampling estimated conditioning pair, not generated results.
"""
import json
from pathlib import Path
import numpy as np
from condmdi_mask import PARENTS

ROOT = Path(__file__).resolve().parent / "text_motion"


def audit(case):
    data = np.load(case / "condition.npz")
    src, dst = data["estimated_source_joints"], data["estimated_canonical_joints"]
    sb = src[:, 1:] - src[:, PARENTS[1:]]
    db = dst[:, 1:] - dst[:, PARENTS[1:]]
    sl = np.linalg.norm(sb, axis=-1, keepdims=True)
    dl = np.linalg.norm(db, axis=-1, keepdims=True)
    assert sl.min() > 1e-6 and dl.min() > 1e-6
    sdir, ddir = sb / sl, db / dl
    u, _, vt = np.linalg.svd(sdir.reshape(-1, 3).T @ ddir.reshape(-1, 3))
    correction = np.diag([1., 1., np.linalg.det(u @ vt)])
    row_rotation = u @ correction @ vt
    direction_error = float(np.abs(sdir @ row_rotation - ddir).max())
    assert direction_error < 1e-4, (case.name, direction_error)
    restored_bones = (ddir @ row_rotation.T) * sl
    # Parent indices are topologically ordered.
    restored = np.zeros_like(src)
    restored[:, 0] = src[:, 0]
    for joint in range(1, 22):
        restored[:, joint] = restored[:, PARENTS[joint]] + restored_bones[:, joint - 1]
    joint_error = float(np.abs(restored - src).max())
    assert joint_error < 1e-4, (case.name, joint_error)
    report = dict(case=case.name, state="conditioning_geometry_retarget_roundtrip_passed",
                  direction_max_error=direction_error, joint_max_error_m=joint_error,
                  source_to_standard_row_rotation=row_rotation.tolist(),
                  source_bone_lengths_m=sl[0, :, 0].tolist(),
                  source_bone_length_temporal_max_error_m=float(np.abs(sl - sl[:1]).max()),
                  scope="Known pre-sampling estimated skeleton pair only; not generated pose retargeting or ground-truth motion",
                  root_policy="This inverse construction explicitly keeps original root; no generated trajectory adopted")
    (case / "retarget_geometry.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    reports = [audit(ROOT / "cases" / c) for c in json.loads((ROOT / "PROTOCOL.json").read_text())["cases"]]
    (ROOT / "retarget_geometry_audit.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps({r["case"]: r["joint_max_error_m"] for r in reports}))
