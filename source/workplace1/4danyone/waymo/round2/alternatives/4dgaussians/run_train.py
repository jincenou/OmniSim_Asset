"""Run the pinned author's optimization recipe with explicit local input adapters."""
from argparse import ArgumentParser
from datetime import datetime
import hashlib
import copy
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
import numpy as np
import torch
from gpu_guard import validate_gpu

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A / "repo"))
import train
from arguments import ModelParams, ModelHiddenParams, PipelineParams, OptimizationParams


def main():
    command = ArgumentParser()
    command.add_argument("--label", choices=["batch_0084_twentyfour", "walk063_rootlocal24", "m02_short61_24", "r07_short61_24", "phone_full", "batch_0021_rootlocal24"], required=True)
    command.add_argument("--run", required=True)
    command.add_argument("--smoke", action="store_true")
    command.add_argument("--objective", choices=["official_rgb", "alpha_l1"], default="official_rgb")
    command.add_argument("--temporal-control", choices=["grid61", "smooth001"])
    requested = command.parse_args()
    if requested.temporal_control:
        assert requested.objective == "official_rgb"
        assert requested.label in ("m02_short61_24", "r07_short61_24", "phone_full", "batch_0021_rootlocal24")
    trainer = train
    trainer_path = A / "repo/train.py"
    if requested.objective == "alpha_l1":
        alpha_proof = json.loads((A / "reports/alpha_preflight.json").read_text())
        assert alpha_proof["state"] == "full_image_soft_alpha_forward_backward_passed"
        trainer_path = A / "variants/alpha_l1/train.py"
        spec = importlib.util.spec_from_file_location("fd_alpha_train", trainer_path)
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
    proof_names = {"batch_0084_twentyfour": "runtime_preflight.json", "walk063_rootlocal24": "runtime_preflight.json",
        "m02_short61_24": "runtime_preflight_short.json", "r07_short61_24": "runtime_preflight_short.json",
        "phone_full": "runtime_preflight_additional.json", "batch_0021_rootlocal24": "runtime_preflight_additional.json"}
    proof_path = A / "reports" / proof_names[requested.label]
    proof = json.loads(proof_path.read_text())
    assert proof["state"] == "camera_time_knn_rasterizer_forward_backward_passed"
    manifest_path = A / "data" / requested.label / "fdanyone.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    record = next(item for item in proof["observations"] if item["label"] == requested.label)
    if "manifest_sha256" in record:
        assert record["manifest_sha256"] == manifest_hash
    device = validate_gpu()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    out = A / "runs" / requested.run
    out.mkdir(parents=True, exist_ok=False)
    parser = ArgumentParser()
    mp, hp, pp, op = ModelParams(parser), ModelHiddenParams(parser), PipelineParams(parser), OptimizationParams(parser)
    args = parser.parse_args([])
    args.source_path = str(A / "data" / requested.label)
    args.model_path = str(out)
    args.render_process = False
    args.data_device = "cpu"
    args.expname = requested.run
    if requested.temporal_control == "grid61":
        args.kplanes_config = copy.deepcopy(args.kplanes_config)
        assert args.kplanes_config["resolution"] == [64, 64, 64, 25]
        args.kplanes_config["resolution"][-1] = 61
    elif requested.temporal_control == "smooth001":
        assert args.time_smoothness_weight == .01
        args.time_smoothness_weight = .001
    if requested.smoke:
        args.coarse_iterations = 10
        args.iterations = 20
    trainer.args = args
    assert trainer.TENSORBOARD_FOUND
    trainer.network_gui.listener.close()
    optimization = op.extract(args)
    optimization.lambda_alpha = .1 if requested.objective == "alpha_l1" else 0.
    config = {"dataset": vars(mp.extract(args)), "hidden": vars(hp.extract(args)),
        "pipeline": vars(pp.extract(args)), "optimization": vars(optimization), "seed": 42,
        "objective": requested.objective, "trainer_source": str(trainer_path),
        "temporal_control": requested.temporal_control,
        "trainer_sha256": hashlib.sha256(trainer_path.read_bytes()).hexdigest(),
        "device": device, "source_commit": subprocess.check_output(["git", "-C", str(A / "repo"), "rev-parse", "HEAD"], text=True).strip(),
        "input_manifest_sha256": manifest_hash, "preflight": str(proof_path),
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": f"Full backend recipe comparison, not representation-only ablation. White-background L1+official grid regularization; full-image alphaL1 weight={optimization.lambda_alpha}; no mask clipping."}
    if requested.temporal_control:
        baseline_name = {"m02_short61_24": "m02_61_recipe30k_v1", "r07_short61_24": "r07_61_recipe30k_v1",
                         "phone_full": "phone_recipe30k_v1", "batch_0021_rootlocal24": "walkturn0021_recipe30k_v1"}[requested.label]
        baseline_path = A / "runs" / baseline_name
        baseline = json.loads((baseline_path / "experiment.json").read_text())
        provenance = json.loads((baseline_path / "code_provenance.json").read_text())
        assert provenance["sha256"] == config["trainer_sha256"]
        assert config["input_manifest_sha256"] == baseline["input_manifest_sha256"]
        dataset = copy.deepcopy(config["dataset"])
        dataset["model_path"] = baseline["dataset"]["model_path"]
        assert dataset == baseline["dataset"]
        assert config["pipeline"] == baseline["pipeline"] and config["seed"] == baseline["seed"]
        hidden = copy.deepcopy(config["hidden"])
        if requested.temporal_control == "grid61":
            hidden["kplanes_config"]["resolution"][-1] = 25
        else:
            hidden["time_smoothness_weight"] = .01
        assert hidden == baseline["hidden"]
        # Older helper records omit this ignored zero field; the RGB trainer never reads it.
        opt_for_compare = copy.deepcopy(config["optimization"])
        assert opt_for_compare.pop("lambda_alpha") == 0 and "lambda_alpha" not in trainer_path.read_text()
        if requested.smoke:
            opt_for_compare["iterations"] = baseline["optimization"]["iterations"]
            opt_for_compare["coarse_iterations"] = baseline["optimization"]["coarse_iterations"]
        baseline_opt = copy.deepcopy(baseline["optimization"])
        if "lambda_alpha" in baseline_opt:
            assert baseline_opt.pop("lambda_alpha") == 0
        assert opt_for_compare == baseline_opt
        config["controlled_baseline"] = str(baseline_path)
        config["baseline_source_provenance"] = provenance
        config["scope"] = "Native temporal control against same trainer,input,canonical initialization,seed,loss,pipeline and budget. Sole explicit hidden setting differs; grid capacity and adaptive Gaussian density may change. No paper-GT or production quality claim."
    (out / "experiment.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    if requested.objective == "official_rgb":
        (out / "code_provenance.json").write_text(json.dumps({"captured_at": config["started_at"],
            "source": str(trainer_path), "sha256": config["trainer_sha256"],
            "snapshot": str(A / "source_snapshots/train61_official_rgb.py")}, indent=2) + "\n")
    saving = sorted({args.coarse_iterations, args.iterations} | ({5000, 15000} if not requested.smoke else set()))
    before = time.monotonic()
    trainer.training(mp.extract(args), hp.extract(args), optimization, pp.extract(args),
        [], saving, [args.iterations], None, -1, args.expname)
    final = out / f"point_cloud/iteration_{args.iterations}"
    for name in ("point_cloud.ply", "deformation.pth", "deformation_table.pth", "deformation_accum.pth", "sampling.json"):
        assert (final / name).is_file() and (final / name).stat().st_size > 0
    sampling = json.loads((final / "sampling.json").read_text())
    for stage, counts in sampling.items():
        values = np.asarray(counts)
        assert values[[0, 8, 16]].sum() == 0
        assert int(values.sum()) == (args.coarse_iterations if stage == "coarse" else args.iterations)
    complete = {"state": "trained_exported_replay_and_metrics_pending", "elapsed_seconds": time.monotonic() - before,
        "device": device, "fine_iterations": args.iterations, "coarse_iterations": args.coarse_iterations,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "ended_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
    (out / "complete.json").write_text(json.dumps(complete, indent=2, allow_nan=False) + "\n")
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
