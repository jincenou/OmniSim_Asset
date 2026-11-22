"""Bidirectional instance masks from one fixed annotation-based context anchor."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import av
import numpy as np
from PIL import Image, ImageDraw
import torch

G = Path(__file__).resolve().parent
B = G / "text_motion/bilateral_context"
sys.path[:0] = [str(G.parent / "segmentation"),
               str(G.parent / "scene20/r20261019/scripts")]
from gpu_guard import validate_gpu
from sam2.build_sam import build_sam2_video_predictor


def main():
    started = time.time()
    torch.set_num_threads(4)
    device = validate_gpu()
    protocol = json.loads((B / "MASK_PROTOCOL.json").read_text())
    source = Path(protocol["source"])
    out = source / "context_segmentation"
    out.mkdir(exist_ok=False)
    frames = out / "frames"
    frames.mkdir()
    mapping_path = source / "inputs/frame_mapping.json"
    assert hashlib.sha256(mapping_path.read_bytes()).hexdigest() == protocol["mapping_sha256"]
    mapping = json.loads(mapping_path.read_text())
    boxes = np.asarray([x["bbox_input_xyxy"] for x in mapping["frames"]], dtype=np.float32)
    # Freeze before looking at new masks: largest annotation box, first on ties.
    anchor = int(np.argmax(np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)))
    assert anchor == protocol["anchor_index"]
    rgb = []
    with av.open(str(source / "inputs/main.mp4")) as con:
        con.streams.video[0].thread_count = 1
        for index, frame in enumerate(con.decode(video=0)):
            image = frame.to_image()
            rgb.append(np.asarray(image))
            image.save(frames / f"{index:05d}.jpg", quality=95, subsampling=0)
    assert len(rgb) == len(boxes) == protocol["frames"]
    masks = np.zeros((len(rgb), *rgb[0].shape[:2]), dtype=np.uint8)
    seen = np.zeros(len(rgb), dtype=bool)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            str(G.parent / "segmentation/sam2.1_hiera_large.pt"),
            device="cuda", apply_postprocessing=False)
        state = predictor.init_state(video_path=str(frames), offload_video_to_cpu=True, offload_state_to_cpu=True)
        predictor.add_new_points_or_box(state, frame_idx=anchor, obj_id=1, box=boxes[anchor])
        # One identity anchor, actual temporal propagation in both directions.
        for reverse in (False, True):
            for index, ids, logits in predictor.propagate_in_video(state, start_frame_idx=anchor, reverse=reverse):
                assert ids == [1]
                value = logits[0, 0].sigmoid().float().cpu().numpy().__mul__(255).round().astype(np.uint8)
                if seen[index]:
                    if not np.array_equal(masks[index], value):
                        raise ValueError("Anchor differs across propagation directions")
                else:
                    masks[index] = value
                    seen[index] = True
    if not seen.all():
        raise ValueError("Missing context masks")
    np.savez_compressed(out / "target_masks.npz", masks=masks)
    indices = np.unique(np.r_[np.rint(np.linspace(0, len(rgb)-1, 11)).astype(int), anchor])
    width, height = 264, 480
    sheet = Image.new("RGB", (width * 6, (height + 24) * 2), "white")
    for panel, index in enumerate(indices):
        image = rgb[index].copy()
        a = masks[index].astype(float)[..., None] / 255 * .35
        image = np.rint(image * (1-a) + np.array([0, 255, 0]) * a).astype(np.uint8)
        tile = Image.fromarray(image)
        tile.thumbnail((width, height))
        x, y = panel % 6 * width, panel // 6 * (height + 24)
        sheet.paste(tile, (x, y + 24))
        ImageDraw.Draw(sheet).text((x+4, y+4), f"slot {index}; anchor={anchor}", fill="black")
    sheet.save(out / "mask_review.jpg", quality=94)
    (out / "report.json").write_text(json.dumps({
        "state": "masks_propagated_review_required", "device": device,
        "anchor_index": anchor, "anchor_box": boxes[anchor].tolist(),
        "selection": "Largest annotation bbox area, earliest tie; frozen before inference",
        "directions": ["forward", "backward"], "frames": len(masks),
        "seconds": time.time()-started,
        "masks_sha256": hashlib.sha256((out / "target_masks.npz").read_bytes()).hexdigest(),
        "visibility_warning": "Instance support only. Not semantic limb visibility or hidden-body ground truth.",
        "quality_approved": False,
    }, indent=2) + "\n")
    shutil.rmtree(frames)  # This invocation's decoded temporary JPEGs only.


if __name__ == "__main__":
    main()
