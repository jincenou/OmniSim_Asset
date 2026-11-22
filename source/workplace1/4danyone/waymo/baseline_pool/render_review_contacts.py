"""Aspect-preserving all-frame review sheets for completed packages only."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scan_native import ROOT, sha, write


def render(package, destination):
    ready = json.loads((package / "READY.json").read_text())
    count = ready["candidate"]["output_slots"]
    destination.mkdir(parents=True, exist_ok=False)
    evidence = []
    for name, source, columns, rows, expected in [
        ("projection", package/ready["input_skeleton_comparison"], 4, 8, count),
        ("native", package/"clip0/observed_native.mp4", 5, 5, ready["candidate"]["native_frame_count"]),
    ]:
        width, height = (256, 256) if name == "projection" else (160, 256)
        cap = cv2.VideoCapture(str(source))
        frames = []
        while True:
            ok, image = cap.read()
            if not ok:
                break
            h, w = image.shape[:2]
            scale = min(width/w, (height-20)/h)
            small = cv2.resize(image, (round(w*scale), round(h*scale)))
            sh, sw = small.shape[:2]
            tile = np.full((height, width, 3), 220, np.uint8)
            x, y = (width-sw)//2, 20+(height-20-sh)//2
            tile[y:y+sh, x:x+sw] = small
            cv2.putText(tile, f"{name} {len(frames)}", (3, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
            frames.append(tile)
        cap.release()
        assert len(frames) == expected, (source, len(frames), expected)
        per_page = rows*columns
        for start in range(0, len(frames), per_page):
            part = frames[start:start+per_page]
            viewed = list(range(start, start+len(part)))
            part += [np.full_like(frames[0], 200)]*(per_page-len(part))
            sheet = np.concatenate([np.concatenate(part[i:i+columns], axis=1)
                                    for i in range(0, per_page, columns)], axis=0)
            output = destination/f"{name}_{start:03d}.jpg"
            assert cv2.imwrite(str(output), sheet)
            evidence.append({"source": str(source), "source_sha256": sha(source),
                             "frames": viewed, "sheet": str(output), "sheet_sha256": sha(output)})
    write(destination/"manifest.json", {"package": str(package), "evidence": evidence,
                                       "aspect_preserved": True, "review_performed": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    render(args.package, args.destination)
