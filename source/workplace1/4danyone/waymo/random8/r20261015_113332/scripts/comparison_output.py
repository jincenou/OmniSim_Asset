"""Publish one complete four-backend comparison, without partial/history media."""
import asyncio
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import uuid

BRANCHES = ("ftgs_base5k", "ftgs_opt15k", "4dgs_grid25", "4dgs_grid61")
VIEWS = (0, 8, 16)
POLICY = "complete-four-branch-v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def complete_results(base):
    paths = {branch: base / "evaluations" / branch / "results.json" for branch in BRANCHES}
    if not all(path.exists() for path in paths.values()):
        return None, [branch for branch, path in paths.items() if not path.exists()]
    results = {branch: json.loads(path.read_text()) for branch, path in paths.items()}
    for branch, result in results.items():
        assert result["branch"] == branch and result["label"] == base.name
        assert result.get("snapshot") is None
        assert result["state"] == "new_process_replayed_evaluated_decoded_visual_review_pending"
        assert all(math.isfinite(value) for value in result["mean"].values())
        assert (paths[branch].parent / "asset_manifest.json").exists()
    assert len({result["device"]["pid"] for result in results.values()}) == 4
    assert len({result["protocol"]["all_input_pairs_sha256"] for result in results.values()}) == 1
    assert len({json.dumps(result["crops"], sort_keys=True) for result in results.values()}) == 1
    return results, []


def comparison_ready(base):
    results, missing = complete_results(base)
    path = base / "comparison/validation.json"
    if missing or not path.exists():
        return False
    validation = json.loads(path.read_text())
    return (set(validation.get("successful_branches", [])) == set(BRANCHES)
            and not validation.get("missing_branches")
            and all((base / "comparison" / f"heldout_view{view:02d}_five.mp4").exists() for view in VIEWS))


def render_view(base, staging, view, results):
    import av
    from PIL import Image, ImageDraw
    count = results[BRANCHES[0]]["videos"][f"view{view:02d}.mp4"]["frames"]
    assert all(result["videos"][f"view{view:02d}.mp4"]["frames"] == count for result in results.values())
    path = staging / f"heldout_view{view:02d}_five.mp4"
    with ExitStack() as stack:
        readers = {}
        for branch in BRANCHES:
            reader = stack.enter_context(av.open(str(base / "evaluations" / branch / f"view{view:02d}.mp4")))
            reader.streams.video[0].thread_count = 1
            readers[branch] = iter(reader.decode(video=0))
        output = stack.enter_context(av.open(str(path), "w"))
        stream = output.add_stream("libx264", rate=Fraction(25))
        stream.width, stream.height, stream.pix_fmt = 1760, 664, "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        stream.codec_context.thread_count = 2
        for frame in range(count):
            images = {branch: next(reader).to_image() for branch, reader in readers.items()}
            panels = [(images[BRANCHES[0]].crop((0, 0, 352, 640)), "Generated reference")]
            panels.extend((images[branch].crop((352, 0, 704, 640)), branch) for branch in BRANCHES)
            sheet = Image.new("RGB", (1760, 664), (25, 25, 25))
            draw = ImageDraw.Draw(sheet)
            for column, (picture, title) in enumerate(panels):
                sheet.paste(picture, (352 * column, 24))
                draw.text((352 * column + 4, 5), title, fill="white")
            if frame in (0, count // 2, count - 1):
                sheet.save(staging / f"heldout_view{view:02d}_frame{frame:03d}_five.jpg", quality=95)
            encoded = av.VideoFrame.from_image(sheet)
            encoded.pts, encoded.time_base = frame, Fraction(1, 25)
            for packet in stream.encode(encoded):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(path)) as reader:
        assert sum(1 for _ in reader.decode(video=0)) == count
    return {"file": str(base / "comparison" / path.name), "frames": count, "sha256": sha(path)}


async def publish(root, label):
    base = root / "samples" / label
    results, missing = complete_results(base)
    directory = base / "comparison"
    directory.mkdir(exist_ok=True)
    if missing:
        save(directory / "status.json", {"state": "waiting_for_all_four_evaluated_branches", "missing_branches": missing,
                                         "media_generated": False, "policy": POLICY})
        return
    signature = hashlib.sha256(json.dumps({branch: sha(base / "evaluations" / branch / "results.json")
                                            for branch in BRANCHES}, sort_keys=True).encode()).hexdigest()
    validation_path = directory / "validation.json"
    count = results[BRANCHES[0]]["videos"]["view00.mp4"]["frames"]
    stills = [directory / f"heldout_view{view:02d}_frame{frame:03d}_five.jpg"
              for view in VIEWS for frame in (0, count // 2, count - 1)]
    if comparison_ready(base):
        old = json.loads(validation_path.read_text())
        if all(Path(video["file"]).exists() and sha(Path(video["file"])) == video["sha256"] for video in old["videos"]) and all(path.exists() for path in stills):
            if old.get("input_results_signature", signature) == signature:
                old.update(policy=POLICY, input_results_signature=signature, retained_history_copies=False,
                           preview_images=[{"file": str(path), "sha256": sha(path)} for path in stills])
                save(validation_path, old)
                save(directory / "status.json", {"state": "complete_four_branch_comparison", "policy": POLICY})
                return
    staging = directory / (".build-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        videos = await asyncio.gather(*(asyncio.to_thread(render_view, base, staging, view, results) for view in VIEWS))
        previews = [{"file": str(directory / path.name), "sha256": sha(path)} for path in staging.glob("*.jpg")]
        assert len(previews) == 9
        for path in staging.iterdir():
            path.replace(directory / path.name)
        save(validation_path, {"policy": POLICY, "input_results_signature": signature,
                               "shared_input_pairs_hash": results[BRANCHES[0]]["protocol"]["all_input_pairs_sha256"],
                               "videos": videos, "preview_images": previews,
                               "successful_branches": list(BRANCHES), "required_branches": list(BRANCHES),
                               "missing_branches": [], "retained_history_copies": False})
        save(directory / "status.json", {"state": "complete_four_branch_comparison", "policy": POLICY})
    finally:
        shutil.rmtree(staging)
    models = [root / "four_dg/runs" / f"{label}_grid{grid}/complete.json" for grid in (25, 61)]
    if all(path.exists() for path in models):
        left, right = [json.loads(path.read_text()) for path in models]
        assert left["sampling_trace_sha256"] == right["sampling_trace_sha256"]
        save(directory / "sampling_pair.json", {"state": "actual33000sample_sequences_identical", "sha256": left["sampling_trace_sha256"]})
