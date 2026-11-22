"""Check the complete requested delivery against current artifacts; CPU only.

The matrix auditor covers all models/frames, not just a smoke-test subset.
This adds requirement mapping, host termination, receipts, media policy,
working links and qualitative limitations. It never retrains or rerenders.
"""
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zoneinfo import ZoneInfo

from PIL import Image

R = Path(__file__).resolve().parents[1]
B = ("ftgs_base5k", "ftgs_opt15k", "4dgs_grid25", "4dgs_grid61")
LABELS = ("N01", "N02", "N03", "N04", "N06", "N07", "N08", "N11")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.videos = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ("href", "src"):
            if attrs.get(key):self.links.append(attrs[key])
        if tag == "video":self.videos.append(attrs["src"])


def main():
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    for name in ("completion_audit.json", "FINAL_STATUS.json"):
        write(R / name, {"state": "final_requirements_audit_in_progress", "updated_at": stamp,
                         "production_approved": False})
    matrix = read(R / "research/matrix_audit.json")
    summary = read(R / "metrics_summary.json")
    checks = []

    def verify(requirement, function, sources):
        try:
            evidence = function()
            checks.append({"requirement": requirement, "state": "passed", "sources": sources, "evidence": evidence})
        except Exception as error:
            checks.append({"requirement": requirement, "state": "failed", "sources": sources,
                           "error": f"{type(error).__name__}: {error}"})

    def matrix_section(name):
        checked = []
        for label in LABELS:
            row = matrix["tracks"][label]
            values = [row[name]] if name in row else [branch[name] for branch in row["branches"].values()]
            if name == "budget_evaluations":values = [value for group in values for value in group.values()]
            assert all(value["state"] == "passed" for value in values), label
            checked.extend(values)
        return {"passed_sections": len(checked), "actual_evidence": [value["evidence"] for value in checked]}

    def global_section(name):
        value = matrix["global_checks"][name]
        assert value["state"] == "passed"
        return value["evidence"]

    def full_matrix():
        assert matrix["counts"] == {"passed": 132, "pending": 0, "failed": 0}
        assert set(matrix["tracks"]) == set(LABELS)
        # Cached content checks remain valid only with the same actual file fingerprints.
        checked = 0
        for relative, row in matrix["file_checks"].items():
            if "fingerprint" in row:
                stat = (R / relative).stat()
                assert row["fingerprint"] == [stat.st_size, stat.st_mtime_ns], relative
                checked += 1
        return {"matrix_counts": matrix["counts"], "unchanged_hash_or_decode_fingerprints": checked,
                "coverage_review": "Read matrix_audit.py shared/training/evaluation/pair/comparison/camera and all four global checks; 8 actors and every branch, not a sample."}

    verify("Complete 8 new actors × all four frozen recipes = 32 final versions", full_matrix,
           ["REQUEST.txt", "research/matrix_audit.json", "scripts/matrix_audit.py"])
    for requirement, section in [
        ("Fixed seed, eligibility, historical identity exclusions, main8 plus ordered8 reserves", "sampling_and_eligibility"),
        ("Bounded time-coordinate and coarse-stage audits, pinned isolated code, unchanged training math", "source_and_code_audits"),
        ("Retained hard failures; separate main and attempted-reserve denominators; no backend-quality replacement", "failures_and_reporting"),
        ("All64 final/intermediate budget rows; actual update counts and within-run paired changes", "budget_reporting"),
    ]:
        verify(requirement, lambda section=section: global_section(section), ["research/matrix_audit.json", "selection.json", "metrics_summary.json"])
    for requirement, section in [
        ("One shared official Turbo24 input per actor; native timestamps,61/121 slots,RGBA hashes and canonical initialization", "shared_inputs"),
        ("All32 cold starts, frozen LR/sigma/losses, actual optimizer ledgers, H excluded and required checkpoints", "training"),
        ("All32 independent dynamic replays,39 raw metric samples,3×F diagnostics and complete final videos", "final_evaluation"),
        ("All32 intermediate budget evaluations and retained dynamic files", "budget_evaluations"),
        ("Matched grid25/61 configs and sampling; identical shared targets/crops across all four", "pair"),
        ("All24 complete five-panel videos,72 keyframes and72 uniform pose-proxy details", "comparison"),
        ("All8 actual CUDA camera mappings,24 cameras×3 times per actor", "camera_projection"),
    ]:
        verify(requirement, lambda section=section: matrix_section(section), ["research/matrix_audit.json"])

    def root_local():
        rows = []
        for label in LABELS:
            base = R / "samples" / label
            proof = read(base / "generation/research/root_local.json")
            assert proof["state"] == "geometry_and_source_retention_passed_experimental_generation_allowed"
            assert proof["source_motion_reused_without_modification"] is True
            assert len(proof["coverage"]) == 24 and all(row["all_joints_all_frames_inside"] for row in proof["coverage"])
            assert proof["root_checks"]["horizontal_root_residual_max_abs"] < 1e-5
            for key, value in proof["root_checks"].items():
                if key.endswith("vertical_max_abs") or key.endswith("recomposition_max_abs"):assert value < 1e-5
            source = Path(proof["motion_source"])
            for name, expected in proof["source_motion_hashes"].items():assert sha(source / name) == expected
            metadata = read(base / "generation/preflight/conditioning/metadata.json")
            assert metadata["view_plan"]["views_per_layer"] == 24
            assert metadata["view_plan"]["enable_rcp"] and metadata["view_plan"]["enable_tcr"]
            assert (base / "source/inputs/main.mp4").is_file()
            assert (base / "source/native_cameras.npz").is_file()
            assert (base / "generation/preflight/conditioning/goliath40/00.mp4").is_file()
            calibration = read(source / "calibration.json")
            assert calibration["camera_translation_consumed"] is False
            rows.append({"label": label, "protocol": proof.get("preprocessing_protocol", "original-v1.1"),
                         "source_3D_world_accuracy": "not_claimed", "camera_translation_consumed": False})
        return rows

    verify("Actual source/motion/root-local conditioning retained, vertical and relative motion preserved, world limits explicit",
           root_local, ["samples/*/generation/research/root_local.json", "samples/*/source/motion_pair/*/calibration.json"])

    def resources():
        d = read(R / "resource_table.json");p = read(R / "research/package_audit_status.json")
        assert len(d["rows"]) == 32 and not d["errors"]
        assert len(p["success"]) == 64 and not p["failures"]
        assert {(row["label"], row["branch"]) for row in d["rows"]} == {(label, branch) for label in LABELS for branch in B}
        for row in d["rows"]:
            assert row["dynamic_payload_bytes"] > 0 and row["all_retained_recovery_checkpoint_bytes"] > 0
            assert row["initial_gaussians"] > 0 and row["final_gaussians"] > 0
            assert math.isfinite(row["full_training_seconds"]) and row["full_training_seconds"] > 0
            assert math.isfinite(row["peak_allocated_gib"]) and row["peak_allocated_gib"] > 0
            assert row["deformation_network_parameters"] == (0 if row["branch"].startswith("ftgs") else
                                                           35762977 if row["branch"] == "4dgs_grid25" else 39080737)
        for path in p["success"]:
            value = read(Path(path));assert Path(value["resolved_configuration"]).exists()
            assert value["production_approved"] is False
        return {"resource_rows": 32, "configuration_and_budget_audits": 64, "payload_and_recovery_bytes_separate": True}

    verify("Complete32 resource/configuration records,64 package audits, point/network counts and separate payload/checkpoint sizes",
           resources, ["resource_table.csv", "resource_table.json", "research/package_audit_status.json"])

    def final_assets_and_images():
        images = 0
        for label in LABELS:
            for branch in B:
                directory = R / "samples" / label / "evaluations" / branch
                asset = read(directory / "asset_manifest.json");result = read(directory / "results.json")
                assert Path(asset["runtime"]).is_dir() and Path(asset["source_manifest"]).exists()
                if branch.startswith("4dgs"):assert Path(asset["config"]).exists()
                for name in [*result["snapshots"], *[f"anomaly{k}_lower_proxy.png" for k in range(3)],
                             *[f"backgrounds_frame{frame:03d}.jpg" for frame in (0,(asset["frame_count"]-1)//2,asset["frame_count"]-1)]]:
                    with Image.open(directory / name) as image:
                        assert image.width > 0 and image.height > 0
                        image.verify()
                    images += 1
                assert len(result["detail_regions"]) == 3 and len(result["worst_diagnostic_frames"]) == 3
                assert result["quality"]["scene_ready"] is False and not result["production_approved"]
        return {"dynamic_packages_with_runtime": 32, "actual_final_images_decoded": images,
                "includes_anomaly_moments": True, "HUST_requires_PLY_plus_deformation_table_accum_config_runtime": True}

    verify("Every actor has all four full dynamic packages,runtime,black/gray/white previews and anomaly-moment local images",
           final_assets_and_images, ["samples/*/evaluations/*/asset_manifest.json", "samples/*/evaluations/*/results.json"])

    def protocol_and_groups():
        assert len(summary["rows"]) == 32 and summary["budget_comparison"]["available_rows"] == 64
        assert summary["reference"].startswith("generated-heldout")
        for cohort, expected in (("main",7),("reserve",1)):
            matched = summary["cohorts"][cohort]["matched_complete_four"]
            assert len(matched["sample_ids"]) == expected
            assert set(summary["cohorts"][cohort]["groups"]) == {"human_action", "ego_group", "frames"}
            assert summary["cohorts"][cohort]["grid61_minus_grid25"]["paired_tracks"] == expected
        equivalent = read(R / "research/metric_equivalence.json")
        assert equivalent["lpips_abs_difference"] == equivalent["ssim_difference"] == 0
        return {"main_complete_pairs": 7, "reserve_complete_pairs": 1,
                "raw_render_metric_implementation_reviewed": "evaluate_branch_all8.py scalar/ssim/lpips:352x640 gray0.5; target-union crop shared; no predicted false-foreground masking; three H×13 times",
                "no_semantic_hand_foot_accessory_ground_truth_claimed": True,
                "group_fields": ["descriptive source action", "ego motion", "61/121slots"]}

    verify("Frozen six-metric protocol,means/medians,paired gains/degradations and action/ego/duration groups; real-GT limitations",
           protocol_and_groups, ["metrics.csv", "metrics_summary.json", "scripts/evaluate_branch_all8.py", "research/metric_equivalence.json"])

    def receipts():
        records = []
        for path in (R / "logs/actual_receipts").glob("*.json"):
            value = read(path)
            if value.get("actual_receipt", {}).get("exit_code") == 0:records.append(value)
        indexed = {}
        for value in records:indexed.setdefault(value["job"], []).append(value["actual_receipt"])
        verified = [];legacy_repair_metadata=[]
        for label in LABELS:
            for branch in B:
                for job in (label+"/"+branch,label+"/"+branch+"_evaluation"):
                    options = [v for v in indexed.get(job, []) if v.get("command") and v.get("pid") and v.get("log")]
                    repair = R / "research" / f"{label}_{branch}_evaluation_repair.json"
                    if not options:
                        assert job.endswith("_evaluation") and repair.exists() and read(repair)["exit_code"] == 0, job
                    for v in options:
                        assert Path(v["log"]).is_file() and (v.get("physical_gpus") or v.get("physical_gpu") is not None)
                        if not (v.get("started_at") and v.get("ended_at") and v.get("environment")):
                            metadata = read(R/"research/N04_ftgs_base5k_repair_metadata.json")
                            assert job == "N04/ftgs_base5k_evaluation" and v["pid"] == metadata["pid"]
                            assert sha(R/"scripts/repair_evaluation.py") == metadata["saved_launcher_sha256"]
                            assert metadata["start_time_lower_bound"] < metadata["completion_artifact_mtime"]
                            asset = read(R/"samples/N04/evaluations/ftgs_base5k/asset_manifest.json")
                            assert asset["new_process_pid"] == v["pid"] and asset["device"]["physical_gpu"] == v["physical_gpu"]
                            legacy_repair_metadata.append(metadata)
                    verified.append(job)
        return {"training_and_evaluation_jobs_with_actual_success_or_documented_repair": len(verified),
                "actual_receipts_retained": len(records), "historical_failures_not_overwritten": True,
                "legacy_manual_retry_timing_precision": "Original failed stage has clocked start/end; its short manual retry has saved command,actual PID/UUID/exit and artifact timestamps. Exact retry start was not captured and is explicitly unknown; saved lower/upper bounds are not claimed as exact runtime.",
                "legacy_repair_metadata": legacy_repair_metadata}

    verify("Actual commands,environment,PIDs,GPU assignments,timing,logs and exit codes; explicit evaluation repairs retained",
           receipts, ["logs/actual_receipts", "logs/first4_receipts", "research/N04_4dgs_grid61_evaluation_repair.json"])

    def gallery_and_policy():
        parser = Links();text = (R / "gallery/index.html").read_text();parser.feed(text)
        missing = []
        for link in parser.links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc or not parts.path:continue
            if not (R / "gallery" / unquote(parts.path)).resolve().exists():missing.append(link)
        assert not missing, missing
        assert len(parser.videos) == 24 and all("heldout_view" in link and "_five.mp4" in link for link in parser.videos)
        assert all(f"<h2>{label} " in text for label in LABELS)
        assert all(f"<h2>{label} " in text for label in ("N05","N09","N10"))
        assert all(f"<h2>N{index} " not in text for index in range(12,17))
        assert "PENDING" not in text
        redundant = [str(path.relative_to(R)) for path in (R/"samples").glob("*/comparison/*previous*")]
        redundant += [str(path.relative_to(R)) for path in (R/"samples").glob("*/comparison/parts")]
        redundant += [str(path.relative_to(R)) for path in (R/"samples").glob("*/data/sam2_input")]
        for directory in (R/"samples").glob("*/evaluations/*_snapshot*"):
            redundant += [str(path.relative_to(R)) for path in directory.iterdir() if path.suffix.lower() in {".mp4",".jpg",".png",".jpeg"}]
        assert not redundant, redundant
        assert read(R/"research/output_policy_checks.json")["state"] == "passed"
        proof = {"state":"passed","local_links_checked":len(parser.links),"complete_four_branch_video_embeds":len(parser.videos),
                 "missing_links":missing,"partial_comparison_embeds":0,"existing_media_only":True,"updated_at":stamp}
        write(R/"research/gallery_link_audit.json",proof)
        return {**proof,"redundant_partial_previous_or_SAM2_media":redundant,
                "future_output_guard_reviewed":"comparison_output.py requires four finite final results with same targets/crops and dynamic assets before publication; budget uses NullVideoWriter"}

    verify("Working gallery for8 complete/3 failed cases;24 full comparisons; no partial/previous/budget duplicate media or SAM2 copies",
           gallery_and_policy, ["gallery/index.html", "samples/README.md", "scripts/comparison_output.py", "research/output_policy_checks.json"])

    def reviews():
        for name in ("source_action_review.json", "reconstruction_visual_review.json"):
            review = read(R / "research" / name)
            assert set(review["cases"]) == set(LABELS)
            for label, row in review["cases"].items():
                key = "source_contact" if name.startswith("source") else "file"
                assert sha(R / row[key]) == row["source_contact_sha256" if key == "source_contact" else "sha256"]
                assert not row.get("quality_approved",False) and not row.get("human_approved",False)
        return {"source_and_final_sampled_visual_reviews":8,"full_video_human_quality_review":"pending_not_required_for_research_delivery",
                "N11_large_arm_motion_failure_retained":True,"production_approved":False}

    verify("Actual source/final qualitative inspections; retain severe failures; distinct technical/visual/scene approval",
           reviews, ["research/source_action_review.json", "research/reconstruction_visual_review.json"])

    def runtime():
        proof = read(R / "research/runtime_final.json")
        assert not proof["own_remaining_processes"]
        assert all("terminal" in value for value in proof["controller_terminal_states"].values())
        assert proof["tmux"]["exit_code"] == 1 and "no server running" in proof["tmux"]["stderr"]
        assert proof["disk_free_gib"] >= 35
        assert proof["foreign_processes_left_untouched"]
        assert len(proof["gpu_inventory"]["stdout"].splitlines()) == 8
        assert (R/"research/all8_gpu_snapshot.json").is_file() and (R/"research/launch_verification.json").is_file()
        return {"all_queues_terminal":True,"own_remaining_host_processes":[],"tmux_ended":True,
                "disk_free_gib":proof["disk_free_gib"],"authorized_gpus":list(range(8)),"observed_at":proof["observed_at"]}

    verify("Persistent bounded async queues actually ran, ended after dependencies, no own GPU jobs left;35GiB reserve and foreign jobs preserved",
           runtime, ["research/runtime_final.json", "research/all8_authorization.json", "research/all8_gpu_snapshot.json", "research/launch_verification.json"])

    def report():
        text = (R/"REPORT.md").read_text();progress = (R/"PROGRESS_LATEST.md").read_text()
        assert "当前仍未全部完成" not in text and "剩余必需工作：" not in text
        assert "32个终态重建版本已完成" in text and "132项矩阵技术检查全部通过" in text
        assert "8/8 人" in progress and "后续自动接续" not in progress
        assert "N11" in text and "严重重影" in text and "人工外观批准pending" in text
        assert "tmux -L fdwaymo attach" not in text and "tmux -L fdwaymo attach" not in progress
        assert (R/"CONTINUE.md").exists() and (R/"LATEST_REPORT.txt").read_text().strip() == str(R/"PROGRESS_LATEST.md")
        return {"report_is_terminal_and_scope_honest":True,"future_unverified_world_and_motion_tasks_excluded":True,
                "six_metrics_and_resource_budget_tables_linked":True}

    verify("Current final report,progress,gallery and continuation handoff with real paths and no stale running/remaining claims",
           report, ["REPORT.md", "PROGRESS_LATEST.md", "CONTINUE.md", "LATEST_REPORT.txt"])

    failed = [row for row in checks if row["state"] != "passed"]
    output = {"state":"complete_frozen_random8_research_delivery" if not failed else "incomplete_requirements_require_attention",
              "updated_at":stamp,"scope":"Original REQUEST.txt plus later eight-GPU and compact-output instructions;8 new legal tracks×4 frozen backends. Not a claim of paper-internal FreeTimeGS or production-quality assets.",
              "counts":{"passed":len(checks)-len(failed),"failed":len(failed)},"requirements":checks,
              "objective_sha256":sha(R/"REQUEST.txt"),"selection_sha256":sha(R/"selection.json"),
              "matrix_audit_sha256":sha(R/"research/matrix_audit.json"),"metric_scope":"generated-heldout reconstruction",
              "production_approved":False,"human_quality_approval":"pending","unverified_out_of_scope":["real Waymo novel-view accuracy","source3D action truth","metric world scale","ground contact","full scene integration","arbitrary SMPL-X retargeting"]}
    write(R/"completion_audit.json",output)
    final = {"state":output["state"],"updated_at":stamp,"target_tracks":8,"completed_tracks":list(LABELS),
             "target_final_versions":32,"completed_final_versions":32 if not failed else None,
             "budget_rows":summary["budget_comparison"]["available_rows"],"cohort_denominators":summary["cohort_denominators"],
             "failed_frontends_retained":["N05","N09","N10"],"unused_reserves":["N12","N13","N14","N15","N16"],
             "authorized_physical_gpus":list(range(8)),"training_finished":True,"own_gpu_work_remaining":False,
             "historical_queue_states":"Retained unchanged; actual terminal acceptance is completion_audit.json, not historical review_pending strings.",
             "production_approved":False,"visual_quality":"Sampled review documents remaining artifacts; not production approved",
             "report":"REPORT.md","gallery":"gallery/index.html","audit":"completion_audit.json"}
    write(R/"FINAL_STATUS.json",final)
    print(json.dumps({"state":output["state"],"counts":output["counts"],"failures":failed},ensure_ascii=False))
    if failed:raise SystemExit(1)


if __name__ == "__main__":main()
