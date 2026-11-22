"""Read-only cross-subject root-motion diagnostic; no new pose fitting."""
import asyncio
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation
from safetensors.torch import load_file

G = Path(__file__).resolve().parent
OUT = G / "root_audit_dev6"
CAM_PY = "/mnt/ojc/miniconda3/envs/4dgaussians/bin/python"


async def extract(sample, source, semaphore):
    async with semaphore:
        target = OUT / (sample + "_boxes.json")
        if target.exists():
            prior = json.loads(target.read_text())
            candidate = json.loads((source / "candidate.json").read_text())
            assert prior["identity"] == candidate["associated_lidar_track_id"]
            assert prior["source"] == candidate["raw_tfrecord"]
            stat = Path(prior["source"]).stat()
            assert prior["raw_stat"] == dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            assert prior["observations"]
            return
        with (OUT / (sample + ".log")).open("x") as log:
            child = await asyncio.create_subprocess_exec(
                CAM_PY, str(G / "audit_identity_boxes.py"),
                "--candidate", str(source / "candidate.json"), "--output", str(target),
                stdout=log, stderr=log,
                env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2"))
            rc = await child.wait()
        if rc:
            raise RuntimeError(f"{sample} annotation read exited {rc}")


def audit(sample, source):
    evidence = G.parent / "frontend_quality_20261020/motion_unmasked" / sample
    maps = np.load(evidence / "measured_camera_maps.npz")
    native = np.load(evidence / "native_cameras.npz")
    absolute_times = maps["relative_exposure_seconds"] + native["image_center_timestamps_s"][0]
    boxes = json.loads((OUT / (sample + "_boxes.json")).read_text())["observations"]
    bt = np.array([x["timestamp_s"] for x in boxes])
    supported = (absolute_times >= bt[0]) & (absolute_times <= bt[-1])
    if not supported.all():
        return dict(state="insufficient_annotation_time_support", valid_frames=int(supported.sum()),
                    total_frames=len(supported), quality_approved=False)
    centers = np.array([x["center_world"] for x in boxes])
    target = np.column_stack([np.interp(absolute_times, bt, centers[:, j]) for j in range(3)])
    state = load_file(str(source / "motion/motion.safetensors"))
    n = len(state["smpl_params_incam.transl"])
    assert n == len(absolute_times)
    body = smplx.create(str(G.parents[1] / "repo/models/body_models"), model_type="smplx",
                        gender="neutral", flat_hand_mean=True, use_pca=False, batch_size=n).eval()
    p = {k: state["smpl_params_incam." + k] for k in ("body_pose", "betas", "global_orient", "transl")}
    # Keep the actual source betas; this diagnoses source geometry, not a new fixed-shape model.
    with torch.inference_mode():
        vertices = body(**p).vertices.numpy().astype(np.float64)
    camera = maps["camera_to_waymo_world"]
    world = np.einsum("tij,tnj->tni", camera[:, :3, :3], vertices) + camera[:, None, :3, 3]
    midpoint = (world.min(1) + world.max(1)) / 2
    delta_error = (midpoint - midpoint[:1]) - (target - target[:1])
    gpos = state["smpl_params_global.transl"].numpy()
    gR = Rotation.from_rotvec(state["smpl_params_global.global_orient"].numpy()).as_matrix()
    cR = Rotation.from_rotvec(state["smpl_params_incam.global_orient"].numpy()).as_matrix()
    rotations = camera[:, :3, :3] @ cR @ gR.transpose(0, 2, 1)
    average = Rotation.from_matrix(rotations).mean()
    residual = (average.inv() * Rotation.from_matrix(rotations)).magnitude()
    path = lambda v: float(np.linalg.norm(np.diff(v, axis=0), axis=1).sum())
    np.savez_compressed(OUT / (sample + "_curves.npz"), time_s=absolute_times,
                        measured_camera_world_mesh_center=midpoint,
                        associated_box_center=target, native_global_root=gpos)
    return dict(
        state="diagnosed_not_ground_truth_pose", frames=n,
        duration_s=float(absolute_times[-1] - absolute_times[0]),
        box_horizontal_path_m=path(target[:, :2]),
        measured_camera_mesh_horizontal_path_m=path(midpoint[:, :2]),
        native_global_horizontal_root_path_m=path(gpos[:, [0, 2]]),
        box_horizontal_displacement_m=float(np.linalg.norm(target[-1, :2] - target[0, :2])),
        mesh_vs_box_relative_horizontal_error_median_m=float(np.median(np.linalg.norm(delta_error[:, :2], axis=1))),
        mesh_vs_box_relative_horizontal_error_max_m=float(np.linalg.norm(delta_error[:, :2], axis=1).max()),
        global_to_world_rotation_residual_deg_median_max=np.degrees([np.median(residual), residual.max()]).tolist(),
        global_up_in_world=(average.as_matrix() @ np.array([0., 1., 0.])).tolist(),
        source_motion_sha256=hashlib.sha256((source / "motion/motion.safetensors").read_bytes()).hexdigest(),
        quality_approved=False)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=args.resume)
    cfg = json.loads((G / "generation_dev6/PROTOCOL.json").read_text())
    sources = {name: Path(cfg["sources"][name]) for name in cfg["subjects"]}
    plan = dict(
        subjects=list(sources), purpose="Check root-trajectory ambiguity across all fixed Dev6, not P18 only",
        outputs="Official box records, source trajectory curves and numerical diagnostics; no generated videos",
        supervision_used=False, body_heading_forced=False,
        limits="Object boxes are not pelvis/joint ground truth; moving body center and annotations can differ. Frame/LiDAR timestamps linearly interpolated inside support; no scan or rolling-shutter truth.")
    if args.resume:
        assert json.loads((OUT / "PLAN.json").read_text()) == plan
        assert not (OUT / "results.json").exists()
    else:
        (OUT / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n")
    semaphore = asyncio.Semaphore(2)
    results = await asyncio.gather(*(extract(k, v, semaphore) for k, v in sources.items()), return_exceptions=True)
    torch.set_num_threads(2)
    report = {}
    for (sample, source), extraction in zip(sources.items(), results):
        if isinstance(extraction, BaseException):
            report[sample] = dict(state="extraction_failed", error=str(extraction), quality_approved=False)
            continue
        try:
            report[sample] = audit(sample, source)
        except Exception as exc:
            report[sample] = dict(state="diagnostic_failed", error=repr(exc), quality_approved=False)
    (OUT / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
