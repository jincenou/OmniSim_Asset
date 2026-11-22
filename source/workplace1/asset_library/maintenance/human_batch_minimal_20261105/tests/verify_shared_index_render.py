"""Compare original and minimal asset packages through the real SplatAD renderer.

Run in the installed Hasim environment; --splatad-python selects the separate
native worker environment. The same trajectory/camera/time and worker are used
for each pair, including the optional contact appearance effect.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

source = Path('/mnt/ojc/workplace1/Hasim_trajectory_assets/src')
spec = importlib.util.spec_from_file_location("hasim", source / "__init__.py",
                                            submodule_search_locations=[str(source)])
module = importlib.util.module_from_spec(spec)
sys.modules["hasim"] = module
spec.loader.exec_module(module)

from hasim.reconstruction.splatad.scene_renderer import SplatADSceneRenderer
from hasim.shared.assets.library_v2 import AssetLibrary
from hasim.shared.insertion import InsertionScenario, SceneState
from hasim.shared.grounding import configure_runtime_grounding
from hasim.shared.trajectory_file import TrajectoryFile


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def difference(first, second):
    if first.shape != second.shape or first.dtype != second.dtype:
        raise AssertionError(f"array shape/dtype changed: {first.shape}/{first.dtype}, {second.shape}/{second.dtype}")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise AssertionError("nonfinite values in equivalence comparison")
    return dict(max_abs=float(np.max(np.abs(first.astype(np.float64) - second.astype(np.float64)), initial=0)),
                exact_equal=bool(np.array_equal(first, second)), shape=list(first.shape), dtype=str(first.dtype))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--splatad-python", type=Path, required=True)
    parser.add_argument("--asset-library", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--minimal", action="append", required=True,
                        help="TRACK_ID=PATH to a standalone one-asset minimal library; repeat for several assets")
    parser.add_argument("--timestamps", nargs="+", type=float, required=True)
    parser.add_argument("--size", type=int, nargs=2, default=[800, 450])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-extensions-dir", type=Path)
    parser.add_argument("--cuda-arch-list", help="Optional native-worker CUDA architecture list; match shared-cache users")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--contact-shadows", action="store_true")
    parser.add_argument("--grounding", choices=["observed_patch", "trajectory"], default="observed_patch")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trajectories = TrajectoryFile.load(args.trajectories)
    library = AssetLibrary(args.asset_library)
    tracks = {track.id: track for track in trajectories.tracks}
    scenarios, identities = [], {}
    for item in args.minimal:
        if "=" not in item:
            raise ValueError("--minimal must be TRACK_ID=PATH")
        track_id, path = item.split("=", 1)
        if track_id in identities or track_id not in tracks:
            raise ValueError(f"duplicate or unknown track: {track_id}")
        minimal = AssetLibrary(path)
        asset = minimal.get(track_id)
        subset = TrajectoryFile((tracks[track_id],))
        bindings = {track_id: asset.id}
        scenarios.append((InsertionScenario(subset, library, bindings=bindings),
                          InsertionScenario(subset, minimal, bindings=bindings)))
        identities[track_id] = dict(asset_id=asset.id, minimal_path=str(Path(path).resolve()))
    timestamps = np.asarray(args.timestamps)
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = dict(package=str(args.scene.resolve()), python=str(args.splatad_python.absolute()),
                  lidar_enabled=False, contact_shadows=args.contact_shadows, device=args.device,
                  output_size=args.size, log_dir=str(output / "logs"), timeout_s=1200)
    if args.torch_extensions_dir:
        config["torch_extensions_dir"] = str(args.torch_extensions_dir.resolve())
    if args.cuda_arch_list:
        config["cuda_arch_list"] = args.cuda_arch_list
    if args.cache_dir:
        config["cache_dir"] = str(args.cache_dir.resolve())
    report = dict(status="running", utc=datetime.now(timezone.utc).isoformat(),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), config=config,
                  asset_library=str(args.asset_library.resolve()),
                  trajectories=str(args.trajectories.resolve()), assets=identities,
                  grounding=args.grounding, samples=[])
    write_json(output / "progress.json", report)
    renderer = None
    started = time.monotonic()
    try:
        renderer = SplatADSceneRenderer(config)
        for original, minimal in scenarios:
            configure_runtime_grounding(original, renderer.runtime, args.grounding)
            configure_runtime_grounding(minimal, renderer.runtime, args.grounding)
        print("MINIMAL_RENDER_READY", flush=True)
        for timestamp in timestamps:
            original_instances, minimal_instances = [], []
            sample = dict(timestamp_s=float(timestamp), asset_fields={}, placements={})
            for original, minimal in scenarios:
                left, right = original.sample(timestamp), minimal.sample(timestamp)
                if len(left.instances) != 1 or len(right.instances) != 1:
                    raise AssertionError("every requested asset must be active at every comparison time")
                a, b = left.instances[0], right.instances[0]
                fields = {field: difference(getattr(a.cloud, field), getattr(b.cloud, field))
                          for field in ("means", "quaternions", "scales", "opacities", "sh", "local_from_appearance")}
                fields["world_from_local"] = difference(a.world_from_local, b.world_from_local)
                if not all(row["exact_equal"] for row in fields.values()) or a.source_time_s != b.source_time_s:
                    raise AssertionError(f"asset samples differ: {a.track_id} at {timestamp}")
                sample["asset_fields"][a.track_id] = fields
                sample["placements"][a.track_id] = dict(
                    original_world_from_local=a.world_from_local.tolist(),
                    minimal_world_from_local=b.world_from_local.tolist(),
                    input_position=tracks[a.track_id].sample(timestamp).position.tolist())
                original_instances.append(a)
                minimal_instances.append(b)
            camera = renderer.runtime.recorded_pose(timestamp)
            original_frame = renderer.render(SceneState(tuple(original_instances)), camera, timestamp)
            minimal_frame = renderer.render(SceneState(tuple(minimal_instances)), camera, timestamp)
            results = {}
            for field in ("rgb", "depth", "alpha", "semantic"):
                left, right = getattr(original_frame, field), getattr(minimal_frame, field)
                if left.keys() != right.keys():
                    raise AssertionError("camera names changed")
                results[field] = {name: difference(value, right[name]) for name, value in left.items()}
            sample["render_fields"] = results
            sample["original_metadata"] = original_frame.metadata
            sample["minimal_metadata"] = minimal_frame.metadata
            report["samples"].append(sample)
            write_json(output / "progress.json", report)
            if not all(row["exact_equal"] for cameras in results.values() for row in cameras.values()):
                raise AssertionError(f"render changed after minimal export at {timestamp}")
            for label, frame in (("original", original_frame), ("minimal", minimal_frame)):
                directory = output / label
                directory.mkdir(exist_ok=True)
                for name, pixels in frame.rgb.items():
                    Image.fromarray(pixels).save(directory / f"t{timestamp:g}_{name}.png")
            print("MINIMAL_RENDER_EQUAL", float(timestamp), list(original_frame.rgb), flush=True)
        report["bindings"] = [dict(original=first.manifest(), minimal=second.manifest())
                              for first, second in scenarios]
        report["grounding_reports"] = [dict(
            original=first.grounding.report() if first.grounding is not None else None,
            minimal=second.grounding.report() if second.grounding is not None else None)
            for first, second in scenarios]
        renderer.close()
        report["worker_exit_code"] = renderer.runtime.process.poll()
        if report["worker_exit_code"] != 0:
            raise AssertionError("native worker did not exit cleanly")
        report.update(status="passed", elapsed_s=time.monotonic() - started)
        write_json(output / "report.json", report)
        write_json(output / "progress.json", dict(status="passed", elapsed_s=report["elapsed_s"]))
        print("MINIMAL_RENDER_PASSED", report["elapsed_s"], flush=True)
    except BaseException as exc:
        report.update(status="failed", error=str(exc), elapsed_s=time.monotonic() - started)
        write_json(output / "report.json", report)
        write_json(output / "progress.json", dict(status="failed", error=str(exc)))
        raise
    finally:
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
