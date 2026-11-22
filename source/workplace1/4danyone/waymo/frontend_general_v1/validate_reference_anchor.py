"""Validate complete G3 video outputs and unchanged RCP evidence; no quality claim."""
import hashlib
import json
from pathlib import Path
import sys
import av
import numpy as np
from PIL import Image

G = Path(__file__).resolve().parent
B = G / "generation_dev6"


def main():
    sample = sys.argv[1]
    root = B / sample / "G3"
    expected = json.loads((root.parent / "pairing.json").read_text())["frames"]
    anchor = json.loads((root / "anchor_report.json").read_text())
    evidence = json.loads((root / "stage_evidence/report.json").read_text())
    reference = json.loads((root.parent / "G2/stage_evidence/report.json").read_text())
    assert evidence["target_denoising_unchanged"] is False
    assert anchor["final_anchor_equality"] and anchor["anchor_camera_ids"] == [4, 9, 14, 19]
    assert anchor["extra_model_forwards"] == anchor["extra_random_draws"] == 0
    assert evidence["rcp_camera_ids"] == reference["rcp_camera_ids"]
    # Shared-device numerics may differ despite identical weights, inputs and seed.
    # Retain that distinction rather than asserting bitwise RCP identity universally.
    exact_rcp = evidence["rcp_latents_sha256"] == reference["rcp_latents_sha256"]
    records = []
    for stage, entries in evidence["stages"].items():
        for view, record in entries.items():
            v = int(view)
            path = (root / "outputs/turbo/videos" if stage == "target" else root / "stage_evidence/rcp") / f"{v:02d}.mp4"
            keyframes = {e["frame"]: e for e in record["keyframes"]}
            diagnostics = []
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                assert (stream.width, stream.height, stream.average_rate) == (704, 1280, 25)
                count = 0
                for i, frame in enumerate(container.decode(video=0)):
                    count += 1
                    if i in keyframes and keyframes[i]["png"]:
                        png = np.asarray(Image.open(root / keyframes[i]["png"]).convert("RGB"))
                        assert hashlib.sha256(png.tobytes()).hexdigest() == keyframes[i]["rgb_sha256"]
                        diff = frame.to_ndarray(format="rgb24").astype(np.float32) - png.astype(np.float32)
                        diagnostics.append(dict(frame=i, codec_mae=float(np.abs(diff).mean())))
                assert count == expected == record["frames"]
            records.append(dict(stage=stage, view=v, frames=count, codec_diagnostics=diagnostics,
                                imposed_anchor=stage == "target" and v in anchor["anchor_camera_ids"]))
    assert len(evidence["stages"]["target"]) == 24 and len(evidence["stages"]["rcp"]) == 6
    result = dict(state="complete_decoding_and_stage_evidence_verified", quality_approved=False,
                  videos=records, rcp_latents_bitwise_equal_G2=exact_rcp,
                  anchor_views=[4, 9, 14, 19], nonanchor_view_count=20,
                  quality_rule="Report anchor and nonanchor outcomes separately; equality with generated reference is imposed, not quality.",
                  motion_ground_truth=False, visual_review="pending")
    (root / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(sample=sample, validated_videos=len(records), rcp_exact=exact_rcp)))


if __name__ == "__main__":
    main()
