"""Verify reusable original Val12 videos; never rewrite original scene assets."""
import argparse
import hashlib
import json
from pathlib import Path
import av

G = Path(__file__).resolve().parent
B = G / "validation12/generation"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample")
    args = parser.parse_args()
    cfg = json.loads((B / "PROTOCOL.json").read_text())
    assert args.sample in cfg["subjects"]
    source = Path(cfg["sources"][args.sample])
    generation = json.loads((source / "generation.json").read_text())
    assert generation["generation"]["seed"] == 42
    assert generation["generation"]["inference_steps"] == 4
    assert generation["output"] == dict(
        fps="25/1", frames_per_video=cfg["frames"][args.sample],
        height=1280, target_views=24, width=704)
    assert generation["model"]["revision"] == cfg["model_revision"]
    assert generation["model"]["turbo_lora"]["sha256"] == cfg["turbo_sha256"]
    manifest = json.loads((source / "manifest.json").read_text())
    checks = []
    for view in range(24):
        path = source / "videos" / f"{view:02d}.mp4"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            assert (stream.width, stream.height, stream.average_rate) == (704, 1280, 25)
            count = sum(1 for _ in container.decode(video=0))
        assert count == cfg["frames"][args.sample]
        checks.append(dict(view=view, frames=count, sha256=digest))
    output = B / "old_checks" / (args.sample + ".json")
    assert not output.exists()
    output.write_text(json.dumps(dict(
        state="old_outputs_decoded_reusable_as_whole_pipeline_baseline",
        source=str(source), sample=args.sample, videos=checks,
        generation_metadata_sha256=hashlib.sha256((source / "generation.json").read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
        quality_approved=False,
        limitation="Original G0 cameras/preprocessing may differ from G2; not a same-camera single-variable comparison."
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
