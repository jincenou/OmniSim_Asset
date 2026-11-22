"""Compare estimated world motion with independent associated annotation boxes."""
import json
from pathlib import Path
import sys
import hashlib
import argparse
import numpy as np
import torch
import smplx
from safetensors.torch import load_file

G = Path(__file__).resolve().parent
B = G / "text_motion/bilateral_context"
E = B / "repair_v1"
sys.path.insert(0, str(G.parent / "frontend_quality_20261020"))
from motion_evidence import resample_measured_camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-boxes", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    meta = json.loads((E / "source.json").read_text())
    source = E / "source"
    maps = np.load(source / "input/measured_cameras.npz")
    cameras = maps["camera_to_waymo_world"]
    t = maps["relative_exposure_seconds"] + meta["relative_origin_exposure_s"]
    data_path = B / "P18/lidar_boxes.json"
    boxes = json.loads(data_path.read_text())["observations"]
    bt = np.array([x["timestamp_s"] for x in boxes])
    if t[0] < bt[0] or t[-1] > bt[-1]:
        raise ValueError("No annotation temporal support; no extrapolation")
    center = np.array([x["center_world"] for x in boxes])
    target = np.column_stack([np.interp(t, bt, center[:, d]) for d in range(3)])
    raw = load_file(str(source / "motion/motion.safetensors"))
    params = {k: raw["smpl_params_incam."+k] for k in ("body_pose", "betas", "global_orient", "transl")}
    variants = {"raw_context": params}
    for branch in ("no_text", "text"):
        for arm in (("old_anchor", "gap_free", "box_assisted") if args.include_boxes else ("old_anchor", "gap_free")):
            d = load_file(str(E / "cases/P18" / branch / arm / "motion_candidates.safetensors"))
            variants[branch+"_"+arm] = dict(body_pose=d["body_pose"], betas=d["betas"],
                global_orient=d["incam.global_orient"], transl=d["incam.transl"])
    model = smplx.create(str(G.parents[1] / "repo/models/body_models"), model_type="smplx",
        gender="neutral", flat_hand_mean=True, use_pca=False, batch_size=len(t)).eval()
    result, curves = {}, {}
    for name, p in variants.items():
        with torch.inference_mode():
            output = model(**p)
        # Body bounding-box center is a diagnostic, not the annotated object's
        # exact center: the annotation can include umbrella/carried objects.
        vertices_cam = output.vertices.numpy().astype(np.float64)
        vertices_world = np.einsum("tij,tnj->tni", cameras[:, :3, :3], vertices_cam) + cameras[:, None, :3, 3]
        lo, hi = vertices_world.min(1), vertices_world.max(1)
        mid = (lo + hi) / 2
        relative_error = (mid - mid[:1]) - (target - target[:1])
        xy_error = np.linalg.norm(relative_error[:, :2], axis=1)
        result[name] = {
            "world_mesh_center_vertical_range_m": float(np.ptp(mid[:, 2])),
            "world_mesh_center_horizontal_path_m": float(np.linalg.norm(np.diff(mid[:, :2], axis=0), axis=1).sum()),
            "relative_to_first_mesh_vs_box_horizontal_error_median_m": float(np.median(xy_error)),
            "relative_to_first_mesh_vs_box_horizontal_error_max_m": float(xy_error.max()),
            "relative_vertical_error_range_m": float(np.ptp(relative_error[:, 2])),
            "camera_depth_range_m": [float(vertices_cam.mean(1)[:, 2].min()), float(vertices_cam.mean(1)[:, 2].max())],
        }
        curves[name+"_mesh_center_world"] = mid
        curves[name+"_mesh_world_min"] = lo
        curves[name+"_mesh_world_max"] = hi
        curves[name+"_camera_depth"] = vertices_cam.mean(1)[:, 2]
        curves[name+"_relative_error"] = relative_error
    curves.update(timestamp_s=t, box_center_world=target)
    stem = "world_with_box_supervision" if args.include_boxes else "world_diagnostic"
    np.savez_compressed(E / ("evaluation/"+stem+".npz"), **curves)
    source_rows = json.loads((source / "input/frame_mapping.json").read_text())["frames"]
    bbox_height = np.array([x["bbox_input_xyxy"][3]-x["bbox_input_xyxy"][1] for x in source_rows])
    report = {
        "scope": (
            "Associated 3D boxes: independent diagnostic for original arms; "
            "supervision-fit diagnostic for box_assisted arms"
            if args.include_boxes
            else "Independent official associated3D boxes, not optimization supervision"
        ),
        "input_annotation_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "comparison": "Mesh AABB center trajectory vs object box center trajectory, fixed first-frame offset removed",
        "annotation_vertical_range_m": float(np.ptp(target[:, 2])),
        "annotation_horizontal_path_m": float(np.linalg.norm(np.diff(target[:, :2], axis=0), axis=1).sum()),
        "annotation_dimensions_at_middle": boxes[int(np.argmin(abs(bt-t[len(t)//2])))]["dimensions_length_width_height"],
        "annotation_points_min_max_in_context": [
            min(x["num_lidar_points_in_box"] for x in boxes if t[0] <= x["timestamp_s"] <= t[-1]),
            max(x["num_lidar_points_in_box"] for x in boxes if t[0] <= x["timestamp_s"] <= t[-1])],
        "correlation_raw_depth_and_inverse_visible_bbox_height": float(np.corrcoef(curves["raw_context_camera_depth"], 1/bbox_height)[0, 1]),
        "results": result,
        "limitations": [
            "Annotations can include umbrella/carried object and be interpolated; center is not pelvis or precise human surface.",
            "Frame/LiDAR timestamps interpolated to camera exposure; no scan-motion correction or hidden-pose truth.",
            "Correlation is diagnostic, not isolated causal proof.",
            "No forcing foot ground contact or body heading from boxes.",
        ],
        "quality_approved": False,
        "supervision_warning": "For box_assisted arms these annotation boxes were optimization supervision; agreement is not independent validation.",
    }
    (E / ("evaluation/"+stem+".json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
