"""Validate actual generated videos and create compact visual previews."""
import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageOps


def decode(path, expected, samples):
    picked = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if (stream.width, stream.height) != (expected["width"], expected["height"]):
            raise ValueError(f"Unexpected resolution: {path}")
        count = 0
        for index, frame in enumerate(container.decode(video=0)):
            if index in samples:
                picked[index] = frame.to_image()
            count += 1
    if count != expected["frames_per_video"]:
        raise ValueError(f"Frame count {count}: {path}")
    return picked


def tile(image, label, width=176, height=320):
    canvas = Image.new("RGB", (width, height + 24), (20, 23, 29))
    picture = ImageOps.contain(image, (width, height))
    canvas.paste(picture, ((width - picture.width) // 2, 24 + (height - picture.height) // 2))
    ImageDraw.Draw(canvas).text((5, 5), label, fill=(240, 240, 240))
    return canvas


def grid(tiles, columns):
    width, height = tiles[0].size
    sheet = Image.new("RGB", (columns * width, ((len(tiles) + columns - 1) // columns) * height), (20, 23, 29))
    for index, item in enumerate(tiles):
        sheet.paste(item, ((index % columns) * width, (index // columns) * height))
    return sheet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    metadata = json.loads((root / "metadata.json").read_text())
    cameras = json.loads((root / "cameras.json").read_text())["cameras"]
    expected = metadata["output"]
    if len(cameras) != expected["target_views"]:
        raise ValueError("Camera count does not match output metadata")
    if expected["target_views"] == 24:
        parallel = metadata["generation"]["parallelism"]
        if parallel["backend"] != "nccl" or set(parallel["used_devices"]) != {"cuda:0", "cuda:1", "cuda:2", "cuda:3"} or len(parallel["workers"]) != 4:
            raise ValueError("Four GPU execution was not confirmed by metadata")
    for name in ("motion.json", "motion.safetensors"):
        if not (root / "gvhmr" / name).is_file() or not (root / "gvhmr" / name).stat().st_size:
            raise ValueError(f"Missing motion artifact {name}")
    samples = (0, expected["frames_per_video"] // 2, expected["frames_per_video"] - 1)
    def validate(camera):
        for field, shape in (("K", (3, 3)), ("camera_to_world", (4, 4))):
            matrix = np.asarray(camera[field])
            if matrix.shape != shape or not np.isfinite(matrix).all():
                raise ValueError(f"Invalid camera matrix: {field}")
        video = (root / camera["video"]).resolve()
        skeleton = (root / camera["skeleton_video"]).resolve()
        if not video.is_relative_to(root) or not skeleton.is_relative_to(root) or not skeleton.is_file():
            raise ValueError("Missing or external video")
        return decode(video, expected, samples)
    frames = list(map(validate, cameras))
    preview = root / "preview"
    preview.mkdir(exist_ok=True)
    labels = list(map(lambda camera: f"View {camera['camera_id']:02d} | yaw {camera['yaw']:.0f}", cameras))
    sheet = grid(list(map(lambda pair: tile(pair[0][samples[1]], pair[1]), zip(frames, labels))), 6)
    sheet.save(preview / "all_views.jpg", quality=92)
    selected = tuple(range(0, len(cameras), max(1, len(cameras) // 6)))[:6]
    temporal = [tile(frames[index][frame], f"V{index:02d} | frame {frame}") for frame in samples for index in selected]
    grid(temporal, 6).save(preview / "time_views.jpg", quality=92)
    containers = list(map(lambda index: av.open(str(root / cameras[index]["video"])), selected))
    try:
        decoders = list(map(lambda container: container.decode(video=0), containers))
        with av.open(str(preview / "multiview.mp4"), "w") as output:
            stream = output.add_stream("libx264", rate=Fraction(expected["fps"]))
            stream.width, stream.height = 768, 980
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "20", "preset": "fast"}
            for synced in zip(*decoders, strict=True):
                tiles = [tile(frame.to_image(), labels[index], 256, 466) for frame, index in zip(synced, selected, strict=True)]
                image = grid(tiles, 3)
                for packet in stream.encode(av.VideoFrame.from_image(image)):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
    finally:
        list(map(lambda container: container.close(), containers))
    result = {"status": "validated", "views": len(cameras), "frames": expected["frames_per_video"], "dimensions": [expected["width"], expected["height"]], "samples": samples, "selected_views": selected, "generation": metadata["generation"], "previews": ["all_views.jpg", "time_views.jpg", "multiview.mp4"]}
    (preview / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
