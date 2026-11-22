"""Verify compact, recoverable input packages; no generation or quality claim."""
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import av
import numpy as np
from PIL import Image

ROOT = Path(os.environ["FDANYONE_SAMPLE_ROOT"])
PACKAGE = ROOT.parent
CONDITIONS = PACKAGE / "conditions"
sys.path[:0] = [os.environ["FDANYONE_RUNTIME"], "/mnt/ojc/workplace1/4danyone/waymo/scripts"]
from fdanyone.motion.result import MotionResult
from fdanyone.skeleton.pipeline import Conditioning
from fdanyone.video import decode_canonical_clip
from prepare import encode


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    count = int(os.environ["FDANYONE_FRAMES"])
    condition = Conditioning.load(CONDITIONS / "preflight/conditioning")
    assert condition.num_frames == count and len(condition.target_skeletons) == 24
    assert not json.loads((CONDITIONS / "preflight/review.json").read_text())["allow_generation"]
    clip = decode_canonical_clip(ROOT / "inputs/main.mp4", num_frames=count,
                                 start_time=0, fps=Fraction(25))
    motion = MotionResult.load(ROOT / "motion_pair/calibrated_moving")
    motion.validate_against_clip(clip)
    native = json.loads((ROOT / "native.json").read_text())
    observed = ROOT / "observed_native.mp4"
    if not observed.exists():
        encode(observed, [Image.open(f["path"]).convert("RGB") for f in native["frames"]], 10)
    expected = {observed: len(native["frames"]), ROOT / "inputs/main.mp4": count,
                ROOT / "motion_pair/projection_comparison.mp4": count, condition.source_video: count}
    expected.update({s.path: count for s in (*condition.target_skeletons, *condition.rcp_skeletons)})
    decoded = []
    for path, frames in expected.items():
        with av.open(str(path)) as container:
            total = sum(1 for _ in container.decode(video=0))
        assert total == frames, (path, total, frames)
        decoded.append({"path": str(path.relative_to(PACKAGE)), "frames": total, "sha256": sha(path)})
    projection = np.load(ROOT / "motion_pair/calibrated_moving/projection.npz")
    assert all(np.isfinite(projection[k]).all() for k in projection.files)
    # Keep the moving crop K independently before removing redundant RIFE work.
    camera = ROOT / "inputs/cameras.npz"
    if camera.is_symlink():
        resolved = camera.resolve(strict=True)
        temporary = camera.with_suffix(".copying")
        shutil.copy2(resolved, temporary)
        camera.unlink()
        temporary.rename(camera)
    assert sha(camera) == json.loads((ROOT / "inputs/frame_mapping.json").read_text())["camera_npz_sha256"]
    removed = []
    for relative in ["native", "segmentation/frames", "motion_pair/runtime", "inputs_rife"]:
        path = ROOT / relative
        if path.exists():
            assert path.is_dir() and not path.is_symlink()
            removed.append(relative)
            shutil.rmtree(path)
    for relative in ["motion_pair/canonical_clip.mp4"]:
        path = ROOT / relative
        if path.is_file():
            removed.append(relative)
            path.unlink()
    manifest = {"state": "prepared_skeleton_and_identity_review_pending",
                "generation_started": False, "quality_accepted": False,
                "candidate": json.loads((PACKAGE / "candidate.json").read_text()),
                "videos": decoded,
                "inputs": "clip0/inputs/main.mp4",
                "input_skeleton_comparison": "clip0/motion_pair/projection_comparison.mp4",
                "conditioning": "conditions/preflight/conditioning",
                "motion": "clip0/motion_pair/calibrated_moving",
                "mask_risks": json.loads((ROOT / "research/batch_gate_mask.json").read_text()),
                "removed_regenerable_work": removed,
                "recovery": "Native PNGs removed after video/conditioning validation. Original frame paths, hashes, crop transforms and cameras remain in native.json/native_cameras.npz; source dataset is read-only.",
                "limitations": ["Local occlusion is not a rejection.",
                               "Detector reprojection agreement is not hidden-joint truth.",
                               "Generated completion has not been run or validated."]}
    (PACKAGE / "READY.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"state": manifest["state"], "verified_videos": len(decoded),
                      "package_bytes": sum(p.stat().st_size for p in PACKAGE.rglob("*")
                                           if p.is_file() and not p.is_symlink())}), flush=True)


if __name__ == "__main__":
    main()
