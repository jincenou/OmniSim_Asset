"""Prepare frozen validation source evidence only; no repair selection or new generation."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

G = Path(__file__).resolve().parent
W = G.parent
B = G / "validation12"
RUN = W / "scene20/r20261019"
PY = "/mnt/ojc/miniconda3/envs/4danyone/bin/python"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    temp = path.with_suffix(".writing")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def audit(sample):
    sys.path.insert(0, str(W / "frontend_quality_20261020"))
    from motion_evidence import resample_measured_camera
    from safetensors.torch import load_file
    protocol = json.loads((B / "EVIDENCE_PROTOCOL.json").read_text())
    old = Path(protocol["sources"][sample])
    target = B / "evidence" / sample
    previous = json.loads((old / "input/native.json").read_text())
    new = json.loads((target / "native.json").read_text())
    mapping = json.loads((old / "input/frame_mapping.json").read_text())
    assert len(previous["frames"]) == len(new["frames"])
    errors = []
    for before, after in zip(previous["frames"], new["frames"], strict=True):
        assert before["raw_source_sha256"] == after["raw_source_sha256"] == sha(Path(after["raw_source_path"]))
        assert before["sha256"] == after["sha256"] == sha(Path(after["path"]))
        assert before["frame_index"] == after["frame_index"]
        for key in ("K_input", "undistorted_to_input", "bbox_input_xyxy", "crop_xyxy"):
            errors.append(float(np.abs(np.asarray(before[key]) - np.asarray(after[key])).max()))
        assert before["image_center_timestamp_s"] == after["image_center_timestamp_s"]
    assert max(errors) < 1e-8
    c = np.load(target / "native_cameras.npz")
    old_c = np.load(old / "input/native_cameras.npz")
    assert set(c.files) == set(old_c.files)
    assert all(np.array_equal(c[key], old_c[key]) for key in c.files)
    q = np.array([frame["source_relative_exposure_seconds"] for frame in mapping["frames"]])
    t = c["image_center_timestamps_s"]
    world = resample_measured_camera(t - t[0], c["image_center_cv_camera_to_world"], q)
    np.savez_compressed(target / "measured_camera_maps.npz", camera_to_waymo_world=world,
                        world_to_camera=np.linalg.inv(world), relative_exposure_seconds=q)
    state = load_file(str(old / "motion/motion.safetensors"))
    n = len(mapping["frames"])
    assert n == protocol["frames"][sample]
    assert len(state["observed_keypoints_2d"]) == n == len(world)
    assert np.isfinite(world).all()
    assert sha(old / "input/model_input.mp4") == protocol["input_sha256"][sample]
    assert sha(old / "motion/motion.safetensors") == protocol["motion_sha256"][sample]
    masks = np.load(old / "input/target_masks.npz")["masks"]
    assert len(masks) == n and masks.ndim == 3
    assert sha(old / "manifest.json") == protocol["manifest_sha256"][sample]
    save(target / "evidence_audit.json", dict(
        sample=sample, state="source_evidence_verified_no_new_method_evaluation",
        baseline=str(old), native_frames=len(t), model_frames=n,
        old_native_crops_reconstructed_byte_identically=True,
        native_camera_arrays_identical=True, max_metadata_error=max(errors),
        source_video_sha256=sha(old / "input/model_input.mp4"),
        source_motion_sha256=sha(old / "motion/motion.safetensors"),
        measured_maps_sha256=sha(target / "measured_camera_maps.npz"),
        known_camera_scope="Exposure-center approximation; interpolated slots not new measured images",
        pipeline_selected=False, new_generation_started=False, quality_approved=False))
    print(json.dumps({"sample": sample, "native_frames": len(t), "model_frames": n}), flush=True)


async def main():
    protocol = json.loads((B / "EVIDENCE_PROTOCOL.json").read_text())
    assert sha(B / "EVIDENCE_PROTOCOL.json") == (B / "EVIDENCE_PROTOCOL.sha256").read_text().strip()
    assert sha(G / "cohort_v1.json") == protocol["cohort_sha256"]
    for path, digest in protocol["code_sha256"].items():
        assert sha(Path(path)) == digest
    assert not (B / "evidence_status.json").exists()
    sem = asyncio.Semaphore(3)
    state = {"state": "preparing_source_evidence", "jobs": {}}
    async def case(sample):
        async with sem:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", FDANYONE_SAMPLE_ID=sample,
                       FDANYONE_SAMPLE_ROOT=str(B / "evidence" / sample),
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
            try:
                for stage, command in (
                    ("crop", [PY, str(RUN / "scripts/m02_prepare.py")]),
                    ("audit", [PY, str(Path(__file__)), "--audit", sample]),
                ):
                    with (B / "logs" / f"{sample}_{stage}.log").open("x") as log:
                        start = time.time()
                        p = await asyncio.create_subprocess_exec(*command, env=env, stdout=log, stderr=log)
                        state["jobs"][sample] = dict(state="running", stage=stage, pid=p.pid)
                        save(B / "evidence_status.json", state)
                        rc = await p.wait()
                    save(B / "logs" / f"{sample}_{stage}.json", dict(exit_code=rc, seconds=time.time() - start))
                    if rc:
                        raise RuntimeError(f"{sample} {stage} exit {rc}")
                state["jobs"][sample] = {"state": "source_evidence_verified"}
            except Exception as error:
                state["jobs"][sample] = dict(state="failed", error=repr(error))
            save(B / "evidence_status.json", state)
    await asyncio.gather(*(case(sample) for sample in protocol["subjects"]))
    state["state"] = "source_evidence_queue_completed"
    state["failures"] = [sample for sample, record in state["jobs"].items() if record["state"] == "failed"]
    save(B / "evidence_status.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit")
    args = parser.parse_args()
    if args.audit:
        audit(args.audit)
    else:
        asyncio.run(main())
