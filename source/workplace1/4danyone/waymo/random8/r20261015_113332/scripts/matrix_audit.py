"""Read-only checks of the frozen eight-person matrix and its actual artifacts.

This writes one run-level audit, never changes models, scores or previews.
Passed sections do not imply the still-incomplete matrix is complete.
"""
import argparse
import asyncio
import copy
import fcntl
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
from zoneinfo import ZoneInfo

import av
import numpy as np
from PIL import Image

R = Path(__file__).resolve().parents[1]
B = ("ftgs_base5k", "ftgs_opt15k", "4dgs_grid25", "4dgs_grid61")
H = (0, 8, 16)
KEYS = ("full_psnr", "ssim", "foreground_psnr", "crop_lpips", "silhouette_iou", "outside_alpha_mean")
LABELS = ("N01", "N02", "N03", "N04", "N06", "N07", "N08", "N11")
OUT = R / "research/matrix_audit.json"


class AwaitingUpdate(Exception):
    pass


def read(path):
    return json.loads(path.read_text())


def save(data):
    temporary = OUT.with_suffix(".writing")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(OUT)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(function, *args):
    try:
        return {"state": "passed", "evidence": function(*args)}
    except FileNotFoundError as error:
        return {"state": "pending", "missing": str(error.filename)}
    except AwaitingUpdate as error:
        return {"state": "pending", "reason": str(error)}
    except Exception as error:
        return {"state": "failed", "error": f"{type(error).__name__}: {error}", "check": function.__name__}


class Auditor:
    def __init__(self, fresh=False):
        self.cache = {} if fresh or not OUT.exists() else read(OUT).get("file_checks", {})

    def hash(self, path, expected=None):
        stat = path.stat()
        key = str(path.relative_to(R))
        fingerprint = [stat.st_size, stat.st_mtime_ns]
        cached = self.cache.get(key, {})
        if cached.get("fingerprint") != fingerprint:
            cached = {"fingerprint": fingerprint, "sha256": sha(path)}
            self.cache[key] = cached
        if expected is not None:
            assert cached["sha256"] == expected, f"Changed file: {key}"
        return cached["sha256"]

    def video(self, path, expected_sha, frames, size, fps=25):
        self.hash(path, expected_sha)
        cached = self.cache[str(path.relative_to(R))]
        expected = [frames, *size, fps]
        if cached.get("decoded") != expected:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                stream.thread_count = 1
                assert (stream.width, stream.height) == tuple(size), str(path)
                assert float(stream.average_rate) == fps
                count = 0
                for index, frame in enumerate(container.decode(video=0)):
                    assert abs(float(frame.pts * frame.time_base) - index / fps) < 1e-5
                    count += 1
                assert count == frames, (str(path), count, frames)
            cached["decoded"] = expected

    def shared(self, candidate):
        label = candidate["sample_id"]
        base = R / "samples" / label
        f = candidate["output_slots"]
        native = read(base / "source/native.json")
        mapping = read(base / "source/inputs/frame_mapping.json")
        assert native["track_id"] == candidate["track_id"] and native["camera_index"] == candidate["camera_index"]
        assert len(native["frames"]) == candidate["native_frame_count"]
        assert [row["frame_index"] for row in native["frames"]] == [row["frame_index"] for row in candidate["observations"]]
        stamps = np.array([row["image_center_timestamp_s"] for row in native["frames"]])
        assert np.all((np.diff(stamps) >= .08) & (np.diff(stamps) <= .12))
        assert len(mapping["frames"]) == f
        exposure = [row["source_relative_exposure_seconds"] for row in mapping["frames"]]
        assert all(right > left for left, right in zip(exposure, exposure[1:]))
        assert exposure[-1] <= stamps[-1] - stamps[0] + 1e-6
        assert mapping.get("speedup", False) is False and not mapping.get("endpoint_padding", False)
        for index, row in enumerate(mapping["frames"]):
            assert row["slot"] == index and abs(row["video_seconds"] - index / 25) < 1e-8
            left, right = row["left_native_index"], row["right_native_index"]
            assert 0 <= left <= right < len(native["frames"])
            assert row["left_source_frame"] == native["frames"][left]["frame_index"]
            assert row["right_source_frame"] == native["frames"][right]["frame_index"]
            assert 0 <= row["interpolation_fraction"] <= 1
        manifest = read(R / "four_dg/data" / label / "fdanyone.json")
        assert manifest["source_data"] == str(base / "data")
        assert manifest["frame_count"] == f and manifest["time_denominator"] == f - 1
        assert manifest["heldout_views"] == list(H)
        assert manifest["train_views"] == [view for view in range(24) if view not in H]
        images = manifest["input_hashes"]
        assert len(images) == 24 * f
        assert {(row["view"], row["frame"]) for row in images} == {(view, frame) for view in range(24) for frame in range(f)}
        stat_rows = []
        for row in images:
            path = base / "data/images" / f"cam{row['view']:02d}" / f"{row['frame']:06d}.png"
            stat = path.stat()
            stat_rows.append([row["view"], row["frame"], stat.st_size, stat.st_mtime_ns])
        fingerprint = hashlib.sha256(json.dumps(stat_rows).encode()).hexdigest()
        key = label + "/all_shared_RGBA"
        manifest_sha = self.hash(R / "four_dg/data" / label / "fdanyone.json")
        if self.cache.get(key) != {"stat_signature": fingerprint, "manifest_sha256": manifest_sha}:
            for row in images:
                path = base / "data/images" / f"cam{row['view']:02d}" / f"{row['frame']:06d}.png"
                assert sha(path) == row["sha256"]
                with Image.open(path) as image:
                    assert image.mode == "RGBA" and image.size == (352, 640)
            self.cache[key] = {"stat_signature": fingerprint, "manifest_sha256": manifest_sha}
        self.hash(base / "data/init.npz", manifest["source_init_sha256"])
        with np.load(base / "data/init.npz") as init, np.load(R / "four_dg/data" / label / "canonical.npz") as canonical:
            assert np.all(init["velocities"] == 0) and np.allclose(init["durations"], 15 / (f - 1))
            assert np.allclose(np.unique(init["times"]), np.arange(0, f, 15) / (f - 1))
            zero = np.isclose(init["times"], 0)
            points = np.column_stack([init["positions"][zero], np.ones(zero.sum())]) @ np.asarray(manifest["source_to_normalized"]).T
            assert np.max(abs(points[:, :3] - canonical["points"])) < 2e-6
        generated = read(base / "generation/research/generated_validation.json")
        assert generated["state"] == "all24_videos_decoded_same_retained_cameras"
        assert sorted(row["view"] for row in generated["videos"]) == list(range(24))
        for row in generated["videos"]:
            assert row["frames"] == f
            self.video(base / "generation/outputs/turbo/videos" / f"{row['view']:02d}.mp4", row["sha256"], f, (704, 1280))
        metadata = read(base / "generation/outputs/turbo/metadata.json")
        assert metadata["generation"]["seed"] == 42 and metadata["generation"]["inference_steps"] == 4
        assert metadata["generation"]["attention_backend"] == "sdpa"
        source = read(base / "source/visualization/manifest.json")
        assert source["native_observations"] == candidate["native_frame_count"]
        for name, row in source["videos"].items():
            self.video(base / "source/visualization" / name, row["sha256"], row["frames"], row["dimensions"], row["fps"])
        return {"shared_manifest_sha256": manifest_sha, "RGBA_files_verified": len(images), "generated_videos_decoded": 24,
                "native_observations": len(native["frames"]), "output_slots": f, "initialization_common_canonical_verified": True}

    def training(self, candidate, branch):
        label, f = candidate["sample_id"], candidate["output_slots"]
        base = R / "samples" / label
        ft = branch.startswith("ftgs")
        directory = base / branch if ft else R / "four_dg/runs" / f"{label}_grid{branch[-2:]}"
        done = read(directory / ("training_complete.json" if ft else "complete.json"))
        config = read(directory / "experiment.json")
        if ft:
            steps = 5000 if branch == B[0] else 15000
            cfg = config["full_config"]
            assert cfg["max_steps"] == cfg["lr_schedule_steps"] == steps and cfg["ckpt_path"] is None
            assert cfg["train_view_ids"] == [view for view in range(24) if view not in H] and cfg["validation_view_ids"] == list(H)
            assert cfg["data_dir"] == str(base / "data") and cfg["end_frame"] == f
            assert done["actual_optimizer_updates"] == steps
            ledger = np.asarray(done["sampling"]["counts_by_camera_and_frame"])
            assert ledger.shape == (24, f) and ledger.sum() == steps and not ledger[list(H)].any()
            initial = read(directory / "initial_parameters.json")
            sigma = (15 if branch == B[0] else 8) / (f - 1)
            assert math.isclose(initial["duration_sigma_min"], sigma, rel_tol=1e-5)
            assert math.isclose(initial["duration_sigma_max"], sigma, rel_tol=1e-5) and initial["zero_velocity"] is True
            self.hash(Path(cfg["init_npz_path"]), config["init_sha256"])
            self.hash(base / "data/init.npz", config["base_init_sha256"])
            milestones = [5000] if steps == 5000 else [5000, 10000, 15000]
            for step in milestones:
                assert (directory / "ckpts" / f"ckpt_{step - 1}.pt").stat().st_size > 0
                assert (directory / f"ckpt_{step - 1}.ftgs.ply").stat().st_size > 0
            self.hash(Path(done["checkpoint"]), done["checkpoint_sha256"])
        else:
            assert done["actual_optimizer_updates"] == 32999 and done["actual_updates_by_stage"] == {"coarse": 3000, "fine": 29999}
            assert config["seed"] == config["sampling_seed"] == 42
            assert config["hidden"]["kplanes_config"]["resolution"][-1] == int(branch[-2:])
            self.hash(R / "four_dg/data" / label / "fdanyone.json", config["input_manifest_sha256"])
            self.hash(R / "source/4dgaussians/train.py", config["trainer_sha256"])
            self.hash(directory / "sampling_trace.npy", done["sampling_trace_sha256"])
            trace = np.load(directory / "sampling_trace.npy")
            assert trace.shape == (33000, 3) and np.sum(trace[:, 0] == 0) == 3000 and np.sum(trace[:, 0] == 1) == 30000
            assert not np.isin(trace[:, 1], H).any() and trace[:, 2].max() < f
            for stage, step in (("coarse", 3000), ("fine", 12000), ("fine", 30000)):
                checkpoint = directory / f"chkpnt_{stage}_{step}.pth.gz"
                proof = read(checkpoint.with_suffix(".json"))
                self.hash(checkpoint, proof["sha256"])
                assert proof["config_sha256"] == config["frozen_config_sha256"] and proof["input_manifest_sha256"] == config["input_manifest_sha256"]
            sampling = read(directory / "point_cloud/iteration_30000/sampling.json")
            for stage, total in (("coarse", 3000), ("fine", 30000)):
                counts = np.asarray(sampling[stage])
                assert counts.shape == (24, f) and counts.sum() == total and not counts[list(H)].any()
        return {"actual_optimizer_updates": done["actual_optimizer_updates"], "initial_gaussians": done["initial_gaussians"],
                "final_gaussians": done["final_gaussians"], "device": done["device"], "independent_cold_start_and_H_exclusion": True}

    def evaluation(self, candidate, branch, snapshot=None):
        label, f = candidate["sample_id"], candidate["output_slots"]
        base = R / "samples" / label
        directory = base / "evaluations" / (branch if snapshot is None else f"{branch}_snapshot{snapshot}")
        result = read(directory / "results.json")
        assert result["label"] == label and result["branch"] == branch and result["snapshot"] == snapshot
        sampled = np.rint(np.linspace(0, f - 1, 13)).astype(int).tolist()
        assert result["sampled_frames"] == sampled and result["heldout_views"] == list(H)
        assert len(result["samples"]) == 39 and len(result["dense_anomaly_samples"]) == 3 * f
        assert {(row["view"], row["frame"]) for row in result["samples"]} == {(view, frame) for view in H for frame in sampled}
        assert {(row["view"], row["frame"]) for row in result["dense_anomaly_samples"]} == {(view, frame) for view in H for frame in range(f)}
        hashes = []
        for row in result["samples"]:
            assert all(math.isfinite(row[key]) for key in KEYS)
            assert row["crop_xyxy"] == result["crops"][str(row["view"])] and row["background"] == .5
            self.hash(base / "data/images" / f"cam{row['view']:02d}" / f"{row['frame']:06d}.png", row["input_sha256"])
            hashes.append([row["view"], row["frame"], row["input_sha256"]])
        assert hashlib.sha256(json.dumps(hashes).encode()).hexdigest() == result["protocol"]["all_input_pairs_sha256"]
        for key in KEYS:
            assert abs(float(np.mean([row[key] for row in result["samples"]])) - result["mean"][key]) < 1e-9
        assert result["protocol"]["resolution"] == [352, 640] and result["protocol"]["background"] == .5
        assert result["production_approved"] is False
        asset = read(directory / "asset_manifest.json")
        assert asset["new_process_pid"] == result["device"]["pid"]
        if branch.startswith("ftgs"):
            self.hash(Path(asset["dynamic_ply"]), asset["ply_sha256"])
            self.hash(Path(asset["checkpoint"]), asset["checkpoint_sha256"])
        else:
            assert {Path(name).name for name in asset["dynamic_files"]} == {"point_cloud.ply", "deformation.pth", "deformation_table.pth", "deformation_accum.pth"}
            for name, expected in asset["dynamic_files"].items():
                self.hash(Path(name), expected)
        if snapshot is None:
            model = base / branch if branch.startswith("ftgs") else R / "four_dg/runs" / f"{label}_grid{branch[-2:]}"
            training = read(model / ("training_complete.json" if branch.startswith("ftgs") else "complete.json"))
            assert training["device"]["pid"] != result["device"]["pid"]
            assert len(result["replay"]["render_errors"]) == 3 and max(result["replay"]["render_errors"]) < 1e-5
            required = {"view00.mp4": (f, (704, 640)), "view08.mp4": (f, (704, 640)), "view16.mp4": (f, (704, 640)),
                        "fixed_action.mp4": (f, (352, 640)), "backgrounds.mp4": (f, (1056, 640)), "orbit_fixed_time.mp4": (72, (352, 640))}
            assert set(result["videos"]) == set(required)
            for name, (count, size) in required.items():
                self.video(directory / name, result["videos"][name]["sha256"], count, size)
            assert len(result["snapshots"]) == 9
            assert all((directory / name).is_file() for name in result["snapshots"])
        return {"results_sha256": self.hash(directory / "results.json"), "mean_recomputed_from_39_rows": True,
                "dense_frames": 3 * f, "new_process_pid": result["device"]["pid"], "dynamic_files_verified": True,
                "six_full_videos_decoded": snapshot is None}

    def pair(self, candidate):
        label = candidate["sample_id"]
        left, right = [R / "four_dg/runs" / f"{label}_grid{grid}" for grid in (25, 61)]
        configs = [read(directory / "experiment.json") for directory in (left, right)]
        normalized = []
        for config in configs:
            config = copy.deepcopy(config)
            for key in ("device", "started_at", "frozen_config_sha256"):
                config.pop(key, None)
            config["dataset"]["model_path"] = "THIS_PERSON_MODEL"
            config["hidden"]["kplanes_config"]["resolution"][-1] = "TIME_GRID"
            normalized.append(config)
        assert normalized[0] == normalized[1], "Grid pair differs outside time-grid and output/device metadata"
        traces = [np.load(directory / "sampling_trace.npy") for directory in (left, right)]
        assert np.array_equal(*traces)
        results = [read(R / "samples" / label / "evaluations" / branch / "results.json") for branch in B]
        assert len({result["protocol"]["all_input_pairs_sha256"] for result in results}) == 1
        assert len({json.dumps(result["crops"], sort_keys=True) for result in results}) == 1
        assert len({result["device"]["pid"] for result in results}) == 4
        initials = [read(R / "samples" / label / branch / "initial_parameters.json") for branch in B[:2]]
        assert initials[0]["point_count"] == initials[1]["point_count"]
        assert {key: value for key, value in initials[0]["parameter_hashes"].items() if key != "durations"} == {key: value for key, value in initials[1]["parameter_hashes"].items() if key != "durations"}
        return {"grid25_grid61_only_time_grid_difference": True, "actual_33000_sampling_rows_identical": True,
                "all_four_share_targets_and_crops": True, "FTGS_same_initial_parameters_except_durations": True}

    def comparison(self, candidate):
        label, f = candidate["sample_id"], candidate["output_slots"]
        directory = R / "samples" / label / "comparison"
        validation = read(directory / "validation.json")
        assert set(validation["successful_branches"]) == set(B) and not validation.get("missing_branches")
        assert not list(directory.glob("*previous*"))
        for row in validation["videos"]:
            self.video(Path(row["file"]), row["sha256"], f, (1760, 664))
        previews = validation.get("preview_images", [])
        assert len(previews) == 9
        for row in previews:
            self.hash(Path(row["file"]), row["sha256"])
        details = read(directory / "details/manifest.json")
        assert details["branches"] == list(B) and len(details["images"]) == 9
        for row in details["images"]:
            self.hash(Path(row["file"]), row["sha256"])
        return {"complete_comparison_videos_decoded": 3, "five_panel_keyframes": 9, "uniform_region_previews": 9}

    def camera(self, candidate):
        proof = read(R / "research" / f"{candidate['sample_id']}_camera_projection.json")
        assert proof["state"] == "new_actual_data_all24_cameras_three_times_cuda_projection_passed"
        assert len(proof["samples"]) == 72 and proof["max_error_px"] < .001
        return {"max_error_px": proof["max_error_px"], "camera_time_samples": 72, "device": proof["device"]}

    def sampling_protocol(self):
        selection = read(R / "selection.json")
        protocol = read(R / "selection_protocol.json")
        pool = read(R / "research/qualified_pool.json")
        assert selection["selection_seed"] == protocol["selection_seed"] == 20261015
        assert protocol["generation_seed"] == protocol["training_seed"] == 42
        assert protocol["model_outputs_seen"] is False and protocol["ego_motion_filter"] is None
        regenerated = sorted(pool, key=lambda row: row["identity_key"])
        random.Random(20261015).shuffle(regenerated)
        keys = [row["identity_key"] for row in pool]
        assert keys == [row["identity_key"] for row in regenerated] and len(keys) == len(set(keys)) == selection["pool_size"]
        assert hashlib.sha256("\n".join(keys).encode()).hexdigest() == selection["pool_order_sha256"]
        chosen = selection["main"] + selection["reserve"]
        assert len(selection["main"]) == len(selection["reserve"]) == 8
        for index, (actual, original) in enumerate(zip(chosen, regenerated[:16])):
            assert all(actual[key] == value for key, value in original.items())
            assert actual["sample_id"] == f"N{index+1:02d}" and actual["ordered_rank"] == index + 1
            assert actual["main_selection_probability"] == 8 / len(pool)
            observations = actual["observations"]
            boxes = np.asarray([row["bbox_xyxy"] for row in observations])
            heights = boxes[:, 3] - boxes[:, 1]
            assert heights.min() >= 80 and np.median(heights) >= 120
            assert boxes[:, :2].min() >= 8 and boxes[:, 2].max() <= 1912 and boxes[:, 3].max() <= 1272
            assert np.mean([row["difficulty"] >= 2 for row in observations]) <= .2
            assert np.all(np.diff([row["frame_index"] for row in observations]) == 1)
            times = np.asarray([row["timestamp_micros"] for row in observations], dtype=np.int64)
            assert np.max(abs((times - times[0]) / 1e6 - np.arange(len(times)) / 10)) <= .001
            assert actual["native_frame_count"] in (25, 49) and actual["output_slots"] == (len(observations) - 1) * 5 // 2 + 1
        exclusions = read(R / "research/exclusion_evidence.json")
        excluded = {row["identity_key"] for row in exclusions}
        assert not set(keys) & excluded
        cross_camera = read(R / "research/exclusion_cross_camera_audit.json")
        assert cross_camera["state"] == "historical_association_cross_camera_exclusions_pass" and not cross_camera["violations"]
        return {"qualified_pool": len(pool), "seed_replayed": 20261015, "ordered_main_and_reserves_reproduced": 16,
                "frozen_source_eligibility_recomputed_for_registered_cases": True, "ego_motion_not_filtered": True,
                "selection_sha256": self.hash(R / "selection.json")}

    def source_and_code_audits(self):
        audit = read(R / "audit.json")
        assert audit["hard_error"] is False
        assert {(row["frames"], row["grid_time_size"]) for row in audit["time_trace"]} == {(61, 25), (61, 61), (121, 25), (121, 61)}
        for row in audit["time_trace"]:
            assert row["dataset_times"] == [0., .5, 1.]
            assert row["sample_arguments"]["align_corners"] is True and row["sample_arguments"]["padding_mode"] == "border"
        for row in audit["coarse_trace"]:
            assert row["configuration"] == {"dataloader": False, "zerostamp_init": False}
            assert set(row["deformation_calls_by_stage"]) == {"fine"}
            assert any(sample["stage"] == "coarse" and sample["frame"] != 0 for sample in row["actual_training_samples"])
            assert not any(sample["view"] in H for sample in row["actual_training_samples"])
        sources = read(R / "source/random8_changes.json")
        for row in sources:
            path = Path(row["isolated"])
            assert R / "source" in path.parents
            self.hash(path, row["sha256"])
            self.hash(Path(row["patch"]), row["patch_sha256"])
        assert (R / "backend").resolve() == R / "source/ftgs"
        for family in ("train_ftgs", "train_4dgs"):
            texts = [(R / "scripts" / f"{family}{suffix}.py").read_text() for suffix in ("", "_priority", "_all8")]
            normalized = [text.replace("from gpu_guard import", "from DEVICE_GUARD import")
                          .replace("from priority_gpu_guard import", "from DEVICE_GUARD import")
                          .replace("from all8_gpu_guard import", "from DEVICE_GUARD import") for text in texts]
            assert len(set(normalized)) == 1, family + " copies differ outside GPU admission"
        return {"both_time_lengths_and_both_grids_audited": True, "coarse_behavior_retained": True,
                "isolated_trainer_and_patch_hashes_verified": True, "device_entry_copies_same_training_math": True,
                "code_audit": "code_audit.md", "source_pin": audit["source_pin"]}

    def failures_and_reporting(self):
        selection = read(R / "selection.json")
        attempted = [row["sample_id"] for row in selection["main"] + selection["reserve"]
                     if (R / "samples" / row["sample_id"] / "source/native.json").exists()]
        assert attempted == [f"N{index:02d}" for index in range(1, 12)]
        failed = [label for label in attempted if label not in LABELS]
        assert failed == ["N05", "N09", "N10"]
        for label in failed:
            assert read(R / "samples" / label / "source/research/batch_gate_mask.json")["passed"] is False
            assert not (R / "samples" / label / "generation/research/generated_validation.json").exists()
        summary = read(R / "metrics_summary.json")
        assert "generated-heldout" in summary["reference"]
        assert summary["cohort_denominators"]["main"]["observed_attempted"] == 8
        assert summary["cohort_denominators"]["reserve"]["observed_attempted"] == 3
        current = {}
        for label in LABELS:
            for branch in B:
                path = R / "samples" / label / "evaluations" / branch / "results.json"
                if path.exists():current[label, branch] = read(path)
        reported = {(row["sample_id"], row["branch"]): row for row in summary["rows"]}
        if set(current) != set(reported):raise AwaitingUpdate("CPU report has not yet adopted the newest evaluation")
        for key, result in current.items():
            assert all(reported[key][metric] == result["mean"][metric] for metric in KEYS)
        for cohort in ("main", "reserve"):
            cases = [row["sample_id"] for row in selection[cohort]]
            complete = [label for label in cases if all((label, branch) in current for branch in B)]
            matched = summary["cohorts"][cohort]["matched_complete_four"]
            if matched["sample_ids"] != complete:raise AwaitingUpdate("Complete-pair summary awaiting refresh")
            for branch in B:
                for metric in KEYS:
                    if complete:
                        values = [current[label, branch]["mean"][metric] for label in complete]
                        assert abs(np.mean(values) - matched["branches"][branch]["mean"][metric]) < 1e-9
                        assert abs(np.median(values) - matched["branches"][branch]["median"][metric]) < 1e-9
        return {"attempted_tracks": len(attempted), "failed_frontends_retained": failed,
                "main_denominator": 8, "reserve_attempt_denominator": 3,
                "reported_final_evaluations_match_actual_files": len(current), "matched_mean_and_median_recomputed": True}

    def budget_reporting(self):
        summary = read(R / "metrics_summary.json")
        if "budget_comparison" not in summary:raise AwaitingUpdate("Budget report awaiting CPU refresh")
        budget = summary["budget_comparison"]
        assert budget["expected_rows_for_eight_tracks"] == 64 and budget["available_rows"] == len(budget["rows"])
        assert len(budget["rows"]) + len(budget["missing"]) == 64
        for row in budget["rows"]:
            path = Path(row["result"])
            result = read(path)
            assert all(row[key] == result["mean"][key] for key in KEYS)
            expected = row["nominal_iterations"] if row["recipe"].startswith("ftgs") else row["nominal_iterations"] - 1
            assert row["actual_updates_represented"] == expected
        for cohort in ("main", "reserve"):
            for pair in budget["cohorts"][cohort]["within_same_run_changes"].values():
                indexed = {(row["sample_id"], row["branch"]): row for row in budget["rows"] if row["cohort"] == cohort}
                for key in KEYS:
                    expected = {label: indexed[label, pair["right"]][key] - indexed[label, pair["left"]][key] for label in pair["sample_ids"]}
                    assert pair["metrics"][key]["per_track_change"] == expected
                    if expected:
                        assert abs(np.mean(list(expected.values())) - pair["metrics"][key]["mean_change"]) < 1e-9
                        assert abs(np.median(list(expected.values())) - pair["metrics"][key]["median_change"]) < 1e-9
        if budget["missing"]:raise AwaitingUpdate(f"{len(budget['missing'])} budget/final rows still depend on unfinished models")
        return {"all_final_and_intermediate_rows_verified": 64, "within_run_paired_deltas_recomputed": True,
                "HUST_native_snapshot_update_count_distinguished": True}


def main(fresh=False):
    auditor = Auditor(fresh)
    selection = read(R / "selection.json")
    candidates = {row["sample_id"]: row for row in selection["main"] + selection["reserve"]}
    output = {"state": "matrix_incomplete_or_requires_attention", "scope": "Actual file hashes, full MP4 decoding, raw metric-row aggregation, sampled inputs, optimizer ledgers, configs, checkpoints, canonical mappings and independent replay records. No new training/rendering; not visual-quality or world-calibration approval.",
              "cache_policy": "Verified file contents/decodes reused only while byte size and mtime_ns are unchanged; --fresh rechecks all bytes and frames.", "tracks": {}, "file_checks": auditor.cache, "production_approved": False}
    output["global_checks"] = {"sampling_and_eligibility": checked(auditor.sampling_protocol),
                               "source_and_code_audits": checked(auditor.source_and_code_audits),
                               "failures_and_reporting": checked(auditor.failures_and_reporting),
                               "budget_reporting": checked(auditor.budget_reporting)}
    for label in LABELS:
        candidate = candidates[label]
        row = {"shared_inputs": checked(auditor.shared, candidate), "branches": {},
               "pair": checked(auditor.pair, candidate), "comparison": checked(auditor.comparison, candidate),
               "camera_projection": checked(auditor.camera, candidate)}
        for branch in B:
            snapshots = [] if branch == B[0] else [5000, 10000] if branch == B[1] else [12000]
            row["branches"][branch] = {"training": checked(auditor.training, candidate, branch),
                                       "final_evaluation": checked(auditor.evaluation, candidate, branch),
                                       "budget_evaluations": {str(step): checked(auditor.evaluation, candidate, branch, step) for step in snapshots}}
        output["tracks"][label] = row
        output["updated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        save(output)
        print(json.dumps({"label": label, "shared_inputs": row["shared_inputs"]["state"], "pair": row["pair"]["state"], "final_evaluations": {branch: value["final_evaluation"]["state"] for branch, value in row["branches"].items()}}), flush=True)
    checks = list(output["global_checks"].values())
    for row in output["tracks"].values():
        checks.extend(row[key] for key in ("shared_inputs", "pair", "comparison", "camera_projection"))
        for branch in row["branches"].values():
            checks.extend([branch["training"], branch["final_evaluation"], *branch["budget_evaluations"].values()])
    output["counts"] = {state: sum(check["state"] == state for check in checks) for state in ("passed", "pending", "failed")}
    if all(check["state"] == "passed" for check in checks):
        output["state"] = "technical_matrix_checks_passed_final_requirements_review_still_required"
    save(output)
    print(json.dumps({"state": output["state"], "counts": output["counts"]}), flush=True)
    return output


async def watch():
    while True:
        result = main()
        if result["state"].startswith("technical_matrix_checks_passed"):
            return
        await asyncio.sleep(120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    lock = (R / "matrix_audit.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    asyncio.run(watch()) if args.watch else main(args.fresh)
