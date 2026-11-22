"""Validate every frame, stream a six-view preview, keep compact reusable outputs."""
import json
from pathlib import Path
import shutil
import sys

import av
from PIL import Image, ImageDraw

from batch import sha, write
from finish_generation import finish


def main():
    base, frames = Path(sys.argv[1]), int(sys.argv[2])
    work = base / ".work"
    gen = work / "generation"
    out = gen / "outputs/turbo"
    row = json.loads((base / "input_reference.json").read_text())
    source = Path(row["package"])
    finish(work, frames)
    # Reuse exact prior input/motion/conditions by reference rather than duplicate them.
    (base / "input").symlink_to(source / "clip0", target_is_directory=True)
    (base / "motion").symlink_to(source / "clip0/motion_pair/calibrated_moving", target_is_directory=True)
    shutil.move(str(out / "videos"), str(base / "videos"))
    shutil.move(str(out / "skeletons"), str(base / "skeletons"))
    for name in ("cameras.json", "root_motion.npz", "root_local.json", "metadata.json"):
        shutil.copy2(out / name, base / name)
    shutil.copy2(gen / "research/generated_validation.json", base / "validation.json")
    shutil.copy2(gen / "generation_device.json", base / "device.json")
    cameras = json.loads((base / "cameras.json").read_text())
    cameras["motion_world_to_local_world_file"] = "root_motion.npz"
    write(base / "cameras.json", cameras)
    views = (0, 4, 8, 12, 16, 20)
    containers = [av.open(str(base / "videos" / f"{v:02d}.mp4")) for v in views]
    for con in containers:
        con.streams.video[0].thread_count = 1
    decoders = [con.decode(video=0) for con in containers]
    selected = {}
    try:
        with av.open(str(base / "preview.mp4"), "w") as output:
            stream = output.add_stream("libx264", rate=25)
            stream.width, stream.height, stream.pix_fmt = 792, 1000, "yuv420p"
            stream.options = {"crf": "20", "preset": "fast", "threads": "2"}
            for t in range(frames):
                panel = Image.new("RGB", (792, 1000), "#222222")
                draw = ImageDraw.Draw(panel)
                for j, (view, decoder) in enumerate(zip(views, decoders)):
                    im = next(decoder).to_image().resize((264, 480))
                    x, y = (j % 3)*264, (j // 3)*500
                    panel.paste(im, (x, y+20))
                    draw.text((x+4, y+3), f"view{view:02d} frame{t}", fill="white")
                if t in (0, frames//2, frames-1):
                    selected[t] = panel.copy()
                for packet in stream.encode(av.VideoFrame.from_image(panel)):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
            assert all(next(decoder, None) is None for decoder in decoders)
    finally:
        for con in containers:
            con.close()
    with av.open(str(base / "preview.mp4")) as con:
        assert sum(1 for _ in con.decode(video=0)) == frames
    contact = Image.new("RGB", (2376, 1000))
    for j, t in enumerate((0, frames//2, frames-1)):
        contact.paste(selected[t], (j*792, 0))
    contact.save(base / "contact.jpg", quality=92)
    assert sha(source / "READY.json") == row["ready_sha256"]
    retained = {str(p.relative_to(base)): sha(p) for p in base.rglob("*")
                if p.is_file() and ".work" not in p.parts and not p.is_symlink()}
    write(base / "manifest.json", {
        "state": "complete24_multiview_videos_decoded", "views": 24, "frames": frames,
        "model": "4DAnyone Turbo", "training": False, "source_package": str(source),
        "ready_sha256": row["ready_sha256"], "candidate": row["candidate"],
        "input_review": row["review"], "visual_quality_review": "pending",
        "quality_approved": False, "retained": retained,
        "references": "input and motion are local symlinks to retained baseline packages; include them when exporting",
        "limits": "Estimated motion and generated appearance; not true hidden-view ground truth or Gaussian assets",
    })
    # Only this new generation's disposable working tree is removed after validation.
    shutil.rmtree(work)


if __name__ == "__main__":
    main()
