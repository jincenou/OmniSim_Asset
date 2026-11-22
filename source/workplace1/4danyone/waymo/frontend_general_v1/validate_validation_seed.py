"""Full video decoding and pre-encode evidence verification, not quality approval."""
import argparse
import hashlib
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image

G = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample")
    parser.add_argument("label", choices=("G0", "G2"))
    args = parser.parse_args()
    root = G / "validation12/second_seed" / args.sample / args.label
    expected = json.loads((root.parent / "conditioning_audit.json").read_text())["frames"]
    metadata = json.loads((root / "outputs/turbo/metadata.json").read_text())
    assert metadata["generation"]["seed"] == 314159
    evidence = json.loads((root / "stage_evidence/report.json").read_text())
    rows = []
    for stage, entries in evidence["stages"].items():
        for view, record in entries.items():
            path = (root / "outputs/turbo/videos" if stage == "target" else root / "stage_evidence/rcp") / f"{int(view):02d}.mp4"
            differences = []
            keyframes = {entry["frame"]: entry for entry in record["keyframes"]}
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                assert stream.width == 704 and stream.height == 1280
                assert stream.average_rate == 25
                count = 0
                for i, frame in enumerate(container.decode(video=0)):
                    count += 1
                    if i in keyframes and keyframes[i]["png"]:
                        png = np.asarray(Image.open(root / keyframes[i]["png"]).convert("RGB"))
                        assert hashlib.sha256(png.tobytes()).hexdigest() == keyframes[i]["rgb_sha256"]
                        diff = frame.to_ndarray(format="rgb24").astype(np.float32) - png.astype(np.float32)
                        differences.append(dict(frame=i, codec_mae=float(np.abs(diff).mean()),
                                                codec_mean_rgb_bias=diff.mean((0, 1)).tolist()))
                assert count == expected == record["frames"], (path, count, expected)
            rows.append(dict(stage=stage, view=int(view), frames=count, codec_diagnostics=differences))
    assert len(evidence["stages"]["target"]) == 24
    assert len(evidence["stages"]["rcp"]) == len(evidence["rcp_camera_ids"])
    report = dict(state="complete_decoding_and_stage_evidence_verified", quality_approved=False,
                  videos=rows, motion_ground_truth=False, visual_review="pending")
    (root / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"sample": args.sample, "label": args.label, "validated_videos": len(rows)}))


if __name__ == "__main__":
    main()
