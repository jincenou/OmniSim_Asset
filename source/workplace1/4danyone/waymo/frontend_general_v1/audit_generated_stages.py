"""Same-camera RCP/target diagnostics using actual pre-encoding frames."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

G = Path(__file__).resolve().parent
B = G / "generation_dev6"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample")
    parser.add_argument("branch", choices=("G1", "G2", "G3"))
    args = parser.parse_args()
    root = B / args.sample / args.branch
    assert json.loads((root / "validation.json").read_text())["state"] == "complete_decoding_and_stage_evidence_verified"
    pairing = json.loads((root.parent / "pairing.json").read_text())
    n = pairing["frames"]
    runtime = G.parent / "round2/moving61_runtime" if n == 61 else G.parent.parent / "repo"
    sys.path.insert(0, str(runtime))
    from fdanyone.skeleton.pipeline import Conditioning
    from fdanyone.skeleton.keypoints import KEYPOINT_NAMES
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    evidence = json.loads((root / "stage_evidence/report.json").read_text())
    payload = json.loads((root / "preflight/conditioning/cameras.json").read_text())
    points = np.load(root / "preflight/conditioning/keypoints_3d.npy")
    conditioning = Conditioning.load(root / "preflight/conditioning")
    tensor = conditioning.load_source_tensor()
    assert tuple(tensor.shape) == (1, 3, n, 1280, 704)
    times = (0, n // 2, n - 1)
    views = (0, 12)
    head_ids = [KEYPOINT_NAMES.index(k) for k in ("nose", "left-eye", "right-eye", "left-ear", "right-ear")]
    torso_ids = [KEYPOINT_NAMES.index(k) for k in ("left-shoulder", "right-shoulder", "right-hip", "left-hip")]
    sheet = Image.new("RGB", (1120, 432 * 3), "#222222")
    rows, hashes, camera_checks = [], {}, []
    def add_image(im, row, col, label):
        im = ImageOps.contain(im, (224, 408))
        sheet.paste(im, (224 * col + (224 - im.width) // 2, row * 432 + 24 + (408 - im.height) // 2))
        ImageDraw.Draw(sheet).text((224 * col + 3, row * 432 + 5), label, fill="white")
    for row, t in enumerate(times):
        source = ((tensor[0, :, t].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).round().astype(np.uint8)
        add_image(Image.fromarray(source), row, 0, f"Actual VAE input t{t}")
        hashes[f"source_tensor_preview_t{t}"] = hashlib.sha256(source.tobytes()).hexdigest()
        for offset, view in enumerate(views):
            index = evidence["rcp_camera_ids"].index(view)
            a, b = payload["rcp_cameras"][index], payload["cameras"][view]
            assert a["camera_id"] == b["camera_id"] == view
            assert a["K"] == b["K"] and a["world_to_camera"] == b["world_to_camera"]
            assert a["skeleton_video"] == b["skeleton_video"]
            if row == 0:
                camera_checks.append(dict(view=view, rcp_index=index, matrices_and_skeleton_identical=True))
            pa = root / f"stage_evidence/rcp/v{index:02d}_t{t:03d}.png"
            pb = root / f"stage_evidence/target/v{view:02d}_t{t:03d}.png"
            ia, ib = Image.open(pa).convert("RGB"), Image.open(pb).convert("RGB")
            for p in (pa, pb):
                hashes[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
            add_image(ia, row, 1 + offset * 2, f"RCP v{view:02d} t{t}")
            add_image(ib, row, 2 + offset * 2, f"Target v{view:02d} t{t}")
            rgb_a, rgb_b = np.asarray(ia).astype(np.float32), np.asarray(ib).astype(np.float32)
            w = np.asarray(a["world_to_camera"])
            cam = points[t] @ w[:3, :3].T + w[:3, 3]
            q = cam @ np.asarray(a["K"]).T
            uv = q[:, :2] / q[:, 2:]
            for region, ids in (("head", head_ids), ("torso", torso_ids)):
                mask = np.zeros((1280, 704), np.uint8)
                if region == "head":
                    lo, hi = uv[ids].min(0), uv[ids].max(0)
                    margin = .2 * (hi - lo)
                    lo, hi = np.floor(lo - margin).astype(int), np.ceil(hi + margin).astype(int)
                    cv2.rectangle(mask, tuple(lo), tuple(hi), 1, -1)
                else:
                    cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(uv[ids]).astype(np.int32)), 1)
                    mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
                selected = mask.astype(bool)
                count = int(selected.sum())
                valid = count >= 100 and bool((cam[ids, 2] > 0).all())
                metrics = None
                if valid:
                    delta = rgb_b[selected] - rgb_a[selected]
                    ya = rgb_a[selected] @ np.array([.2126, .7152, .0722])
                    yb = rgb_b[selected] @ np.array([.2126, .7152, .0722])
                    metrics = dict(rcp_luma_mean=float(ya.mean()), target_luma_mean=float(yb.mean()),
                                   target_minus_rcp_luma=float((yb - ya).mean()),
                                   rgb_mae_between_generated_stages=float(np.abs(delta).mean()))
                rows.append(dict(view=view, frame=t, region=region, roi_pixels=count, valid=valid, metrics=metrics))
    output = root / "stage_comparison.jpg"
    assert not output.exists()
    sheet.save(output, quality=95)
    report = dict(
        sample=args.sample, branch=args.branch, camera_checks=camera_checks, hashes=hashes, rows=rows,
        rules={"frames": list(times), "views": list(views), "head": "projected 5 facial keypoint bbox with 20% extent margin",
               "torso": "projected shoulder/hip convex hull eroded by3px", "min_area_pixels": 100},
        scope="Same-camera generated-stage changes, not reconstruction/identity/color accuracy against true unseen views",
        limitations=["Projected skeleton ROI is not verified semantic segmentation; pixels may include misaligned anatomy/background.",
                     "Source view is generally different; no source-versus-new-view pixel accuracy reported.",
                     "No brightness threshold or automatic front/back classifier. Manual view coverage must be recorded separately."],
        quality_approved=False)
    (root / "stage_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"sample": args.sample, "branch": args.branch, "rows": len(rows), "image": str(output)}))


if __name__ == "__main__":
    main()
