"""Read-only audit of actual temporal evidence reaching the motion prior."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def spans(flags, times):
    """Inclusive sampled endpoints; these are evidence runs, not truth labels."""
    changes = np.diff(np.r_[False, flags, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1) - 1
    return [
        {
            "first_seconds": float(times[a]),
            "last_seconds": float(times[b]),
            "samples": int(b - a + 1),
            "endpoint_span_seconds": float(times[b] - times[a]),
        }
        for a, b in zip(starts, stops)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = np.load(args.condition)
    times = data["time_20fps"]
    supported = data["supported_joints"].astype(bool)
    mask = data["observed_mask"].astype(bool)
    assert len(times) == len(mask) == len(supported)
    assert np.all(np.diff(times) > 0)
    assert np.all(data["condition"][~mask] == 0)
    result = {
        "input": str(args.condition.resolve()),
        "sha256": hashlib.sha256(args.condition.read_bytes()).hexdigest(),
        "scope": "Existing frozen input only; no resampling, training or mask changes.",
        "support_is_not_3d_truth": True,
        "quality_approved": False,
        "unknown_condition_exactly_zero": True,
        "root_condition_runs": spans(mask[:, :4].all(axis=1), times),
        "joints": {},
    }
    for name, joint in (
        ("left_knee", 4), ("right_knee", 5),
        ("left_ankle", 7), ("right_ankle", 8),
    ):
        position = mask[:, 4 + (joint - 1) * 3:4 + joint * 3].all(axis=1)
        rotation = mask[:, 67 + (joint - 1) * 6:67 + joint * 6].all(axis=1)
        velocity = mask[:, 193 + joint * 3:193 + (joint + 1) * 3].all(axis=1)
        assert not np.any(position & ~supported[:, joint])
        result["joints"][name] = {
            "image_support_runs": spans(supported[:, joint], times),
            "position_condition_runs": spans(position, times),
            "rotation_condition_runs": spans(rotation, times),
            "velocity_condition_runs": spans(velocity, times),
            "image_supported_but_position_disabled_runs": spans(
                supported[:, joint] & ~position, times
            ),
        }
    result["interpretation"] = (
        "The absolute-root representation conservatively requires root support. "
        "Image-supported joints can therefore be absent from actual 3D conditions. "
        "Do not simply unmask them: uncertain root makes their absolute 3D values "
        "unreliable. Preserve image evidence separately through calibrated 2D "
        "constraints. Missing pre-occlusion ankle observations are not a complete "
        "observed gait cycle. This audit does not establish a cause or a repair."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "quality_approved": False}))


if __name__ == "__main__":
    main()
