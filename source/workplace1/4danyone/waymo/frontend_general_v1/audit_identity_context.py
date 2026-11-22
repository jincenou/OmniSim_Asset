"""Inspect real, identity-linked observations; never infer visibility from boxes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.cache.read_text())
    tracks = [t for t in data["tracks"]
              if t.get("associated_lidar_track_id") == args.identity]
    if not tracks:
        raise ValueError("No matched official identity")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "identity": args.identity,
        "cache": str(args.cache),
        "cache_sha256": hashlib.sha256(args.cache.read_bytes()).hexdigest(),
        "scope": "All annotated observations in each matched camera, actual RGB.",
        "time_warning": "Cached frame timestamps are not calibrated camera exposure times.",
        "visibility_warning": "Bounding boxes and identity associations do not establish visible limbs.",
        "tracks": [],
    }
    for track in tracks:
        observations = track["observations"]
        camera = track["camera_index"]
        pages = []
        rows = []
        for start in range(0, len(observations), 42):
            chunk = observations[start:start + 42]
            sheet = Image.new("RGB", (7 * 176, math.ceil(len(chunk) / 7) * 212), "white")
            for index, obs in enumerate(chunk):
                frame = obs["frame_index"]
                path = args.images / f"{frame:03d}_{camera}.jpg"
                with Image.open(path) as source:
                    source = source.convert("RGB")
                    x0, y0, x1, y1 = obs["bbox_xyxy"]
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    height = max(y1 - y0, 1)
                    # Retain foreground occluders, unlike white-background model inputs.
                    side = max(height * 2.1, (x1 - x0) * 2.1, 80)
                    crop_box = (round(cx - side / 2), round(cy - side / 2),
                                round(cx + side / 2), round(cy + side / 2))
                    crop = source.crop(crop_box)
                    draw = ImageDraw.Draw(crop)
                    draw.rectangle((x0-crop_box[0], y0-crop_box[1],
                                    x1-crop_box[0], y1-crop_box[1]), outline="lime", width=1)
                    tile = ImageOps.contain(crop, (172, 172))
                column, row = index % 7, index // 7
                sheet.paste(tile, (column * 176, row * 212 + 36))
                ImageDraw.Draw(sheet).text(
                    (column * 176 + 2, row * 212 + 2),
                    f"cam{camera} f{frame}\nbox height {height:.0f}px", fill="black")
                rows.append({**obs, "path": str(path),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "crop_xyxy": crop_box, "visibility": "not_automatically_labeled"})
            name = f"camera{camera}_{chunk[0]['frame_index']:03d}_{chunk[-1]['frame_index']:03d}.jpg"
            sheet.save(args.output / name, quality=93)
            pages.append(name)
        report["tracks"].append({
            "camera": camera, "track_id": track["track_id"],
            "count": len(observations), "pages": pages, "observations": rows,
        })
    (args.output / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
