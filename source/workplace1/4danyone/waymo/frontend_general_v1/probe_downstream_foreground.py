"""Exact generated-frame foreground numeric probe; no SAM or training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import av
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from transformers import AutoModelForImageSegmentation

G = Path(__file__).resolve().parent
D = G / "validation12/downstream"
REPO = G.parents[1] / "repo"
sys.path.insert(0, str(REPO))
from fdanyone.assets import resolve_foreground_model
from fdanyone.config import FOREGROUND


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    info = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
        "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True).strip().split(", ")
    assert int(info[1]) >= 12000
    torch.cuda.init()
    assert torch.cuda.device_count() == 1
    assert str(torch.cuda.get_device_properties(0).uuid) == info[0].removeprefix("GPU-")
    model = AutoModelForImageSegmentation.from_pretrained(
        str(resolve_foreground_model(REPO / "models").resolve()),
        local_files_only=True, trust_remote_code=True).eval().half().cuda()
    transform = transforms.Compose([transforms.Resize(FOREGROUND.image_size), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    rows = []
    for label in ("P33_G0", "P09_G0", "P09_G2", "P33_G2"):
        path = D / "sources" / label / "videos/00.mp4"
        images = []
        with av.open(str(path)) as container:
            container.streams.video[0].thread_count = 1
            for index, frame in enumerate(container.decode(video=0)):
                if index in (0, 15, 30, 45):
                    images.append(frame.to_image().resize((352, 640), Image.Resampling.LANCZOS))
                if index == 45:
                    break
        assert len(images) == 4
        x = torch.stack([transform(im) for im in images]).half().cuda()
        with torch.inference_mode():
            logits = model(x)[-1]
            predictions = logits.sigmoid().cpu()
        finite = torch.isfinite(logits).flatten(1).sum(1).cpu().tolist()
        mask = [np.asarray(to_pil_image(p).resize(im.size).convert("L")) for im, p in zip(images, predictions)]
        rows.append({
            "label": label, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "frames": [0, 15, 30, 45], "logit_elements_per_frame": logits[0].numel(),
            "finite_logits_per_frame": finite,
            "mask_pixels_over127": [int((m > 127).sum()) for m in mask],
            "mask_min_max": [[int(m.min()), int(m.max())] for m in mask],
        })
        print(json.dumps(rows[-1]), flush=True)
    args.output.write_text(json.dumps({
        "gpu": args.gpu, "uuid": info[0], "dtype": "float16", "batch": 4,
        "rows": rows, "quality_approved": False,
        "scope": "Numeric pre-SAM diagnostic on unchanged frames. No output or thresholds modified.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
