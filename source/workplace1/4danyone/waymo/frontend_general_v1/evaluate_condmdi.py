"""Fixed hidden-pixel evaluation. Inputs here are never model conditions."""
import csv
import hashlib
import json
from pathlib import Path
import av
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from safetensors.torch import load_file
from evaluate_occlusion import interpolate_params, _gvhmr_geometry_context, REPO, EDGES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "text_motion"
OCC = ROOT / "occlusion_v1"


def main():
    torch.set_num_threads(2)
    assert json.loads((OUT / "alignment_status.json").read_text())["state"] == "finished_needs_evaluation"
    protocol = json.loads((OUT / "RETARGET_PROTOCOL.json").read_text())
    gate = json.loads((ROOT / "protocol_v2.json").read_text())["gates"]
    rows, verdicts, links = [], [], []
    with _gvhmr_geometry_context(REPO / "third_party/GVHMR"):
        from hmr4d.utils.smplx_utils import make_smplx
        model = make_smplx("supermotion_coco17").eval()
        for case_name in protocol["cases"]:
            case = OCC / case_name
            meta = json.loads((case / "case.json").read_text())
            src = Path(meta["original_source"])
            native = json.loads((src / "native.json").read_text())
            camera = np.load(src / "native_cameras.npz")
            ref = np.load(OCC / "evaluation_reference" / meta["sample"] / "native_detections.npz")
            kp, boxes = ref["keypoints"], ref["boxes"]
            native_t = ref["native_timestamps"] - ref["native_timestamps"][0]
            masks = np.load(case / "evidence_masks.npz")
            rect, redacted = masks["native_rectangles"], masks["native_redacted"]
            xy = kp[..., :2]
            inside = ((xy[..., 0] >= rect[:, None, 0]) & (xy[..., 0] < rect[:, None, 2]) &
                      (xy[..., 1] >= rect[:, None, 1]) & (xy[..., 1] < rect[:, None, 3]))
            in_box = ((xy[..., 0] >= boxes[:, None, 0]) & (xy[..., 0] <= boxes[:, None, 2]) &
                      (xy[..., 1] >= boxes[:, None, 1]) & (xy[..., 1] <= boxes[:, None, 3]))
            confident = (kp[..., 2] >= .5) & in_box
            hidden = confident & inside & redacted[:, None]
            visible = confident & ~(inside & redacted[:, None])
            legs = np.isin(np.arange(17), [11, 12, 13, 14, 15, 16])[None]
            height = boxes[:, 3] - boxes[:, 1]
            raw = load_file(str(case / "rgb/motion/motion.safetensors"))
            o2 = load_file(str(case / "rgb/optimized/motion_candidates.safetensors"))
            t = o2["times_seconds"].numpy()
            in_range = (native_t >= t[0]) & (native_t <= t[-1])
            q = native_t[in_range]
            params0 = {k: raw["smpl_params_incam." + k] for k in ["body_pose", "betas", "global_orient", "transl"]}
            params0["betas"] = o2["betas"]
            variants = {"O0": params0}
            paths = {
                "O2": case / "rgb/optimized/motion_candidates.safetensors",
                "RoHM": case / "rgb/optimized_rohm/motion_candidates.safetensors",
                "no_text": OUT / "cases" / case_name / "no_text/aligned/motion_candidates.safetensors",
                "text": OUT / "cases" / case_name / "text/aligned/motion_candidates.safetensors",
            }
            for name, path in paths.items():
                state = load_file(str(path))
                np.testing.assert_array_equal(state["times_seconds"].numpy(), t)
                np.testing.assert_array_equal(state["betas"].numpy(), o2["betas"].numpy())
                variants[name] = dict(body_pose=state["body_pose"], betas=state["betas"],
                                      global_orient=state["incam.global_orient"], transl=state["incam.transl"])
            metrics, projections = {}, {}
            scale = json.loads((case / "rgb/optimized/report.json").read_text())["body_scale_m"]
            for name, params in variants.items():
                with torch.inference_mode():
                    points = model(**interpolate_params(params, t, q)).numpy()
                    points25 = model(**params).numpy()
                assert np.isfinite(points).all() and (points[..., 2] > 0).all()
                h = np.einsum("tij,tnj->tni", camera["K_input"][in_range], points)
                uv = h[..., :2] / h[..., 2:]
                projections[name] = uv
                error = np.linalg.norm(uv - xy[in_range], axis=-1) / height[in_range, None]
                m = {}
                for region, mask in [("hidden", hidden), ("hidden_legs", hidden & legs), ("visible", visible)]:
                    v = error[mask[in_range]]
                    m[region] = dict(n=int(len(v)), nme_mean=float(v.mean()) if len(v) else None,
                                     nme_p95=float(np.percentile(v, 95)) if len(v) else None)
                rotation = Rotation.from_rotvec(params["global_orient"].numpy()).as_matrix()
                local = np.einsum("tji,tnj->tni", rotation, points25 - params["transl"].numpy()[:, None])
                local -= local[:, 11:13].mean(1, keepdims=True)
                dt = np.diff(t)
                velocity = np.diff(local, axis=0) / dt[:, None, None]
                accel = np.diff(velocity, axis=0) / ((dt[1:] + dt[:-1]) / 2)[:, None, None]
                m["estimated_local_range"] = float(np.sqrt(np.mean(np.ptp(local, axis=0) ** 2)) / scale)
                m["estimated_local_accel"] = float(np.sqrt(np.mean(accel ** 2)) / scale)
                metrics[name] = m
                rows.append(dict(case=case_name, sample=meta["sample"], method=name,
                                 **{f"{r}_{k}": v for r in ["hidden", "hidden_legs", "visible"] for k, v in m[r].items()},
                                 estimated_local_range=m["estimated_local_range"],
                                 estimated_local_accel=m["estimated_local_accel"]))
            # Confirm unchanged comparators and denominators against the existing evaluation.
            old_metrics = json.loads((case / "evaluation/metrics.json").read_text())["metrics"]
            for now, old in [("O0", "rgb_O0"), ("O2", "rgb_O2"), ("RoHM", "rgb_O3")]:
                for region in ["hidden", "hidden_legs", "visible"]:
                    assert metrics[now][region]["n"] == old_metrics[old][region]["n"]
                    for key in ["nme_mean", "nme_p95"]:
                        np.testing.assert_allclose(metrics[now][region][key], old_metrics[old][region][key], atol=1e-7, rtol=0)
            for method in ["no_text", "text"]:
                m = metrics[method]
                for control in ["O0", "O2", "RoHM"] + (["no_text"] if method == "text" else []):
                    c = metrics[control]
                    hidden_gain = 1 - m["hidden"]["nme_mean"] / c["hidden"]["nme_mean"]
                    mean_diff = m["visible"]["nme_mean"] - c["visible"]["nme_mean"]
                    p95_diff = m["visible"]["nme_p95"] - c["visible"]["nme_p95"]
                    amplitude_ratio = m["estimated_local_range"] / c["estimated_local_range"]
                    verdicts.append(dict(
                        case=case_name, method=method, control=control,
                        hidden_nme_relative_reduction=hidden_gain,
                        hidden_nme_improvement_10pct=hidden_gain >= gate["heldout_occlusion_nme_relative_reduction_min"],
                        visible_mean_nme_change=mean_diff, visible_p95_nme_change=p95_diff,
                        visible_protection=mean_diff <= gate["visible_detector_nme_mean_max_increase_absolute"] and
                                           p95_diff <= gate["visible_detector_nme_p95_max_increase_absolute"],
                        estimated_local_amplitude_ratio=amplitude_ratio,
                        amplitude_proxy_in_range=gate["motion_amplitude_ratio_min"] <= amplitude_ratio <= gate["motion_amplitude_ratio_max"],
                        step_frequency=None, stationary_high_frequency=None, full_temporal_visual_review=None,
                        quality_passed=False,
                        limitations="Amplitude/acceleration are estimate diagnostics, not hidden 3D truth. Missing checks do not pass."))
            output = OUT / "cases" / case_name / "evaluation"
            output.mkdir(exist_ok=False)
            columns = ["native heldout", "occluded input", "O2", "RoHM", "no_text", "text"]
            w, h = native["size"]
            panels = []
            for qi, i in enumerate(np.flatnonzero(in_range)):
                original = Image.open(native["frames"][i]["path"]).convert("RGB")
                panel = Image.new("RGB", (w * 6, h + 24), "#222222")
                labels = ImageDraw.Draw(panel)
                for ci, name in enumerate(columns):
                    im = Image.open(case / "rgb/native" / f"{i:05d}.png").convert("RGB") if ci == 1 else original.copy()
                    draw = ImageDraw.Draw(im)
                    if ci >= 2:
                        uv = projections[name][qi]
                        for a, b in EDGES:
                            draw.line([tuple(uv[a]), tuple(uv[b])], fill="lime", width=2)
                        for j, (x, y, _) in enumerate(kp[i]):
                            if confident[i, j]:
                                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="red" if hidden[i, j] else "yellow")
                    panel.paste(im, (ci * w, 24))
                    labels.text((ci * w + 3, 4), f"{name} n{i}", fill="white")
                panels.append(panel)
            with av.open(str(output / "comparison.mp4"), "w") as file:
                stream = file.add_stream("libx264", rate=10)
                stream.width, stream.height, stream.pix_fmt = w * 6, h + 24, "yuv420p"
                stream.options = {"crf": "19", "threads": "2"}
                for panel in panels:
                    for packet in stream.encode(av.VideoFrame.from_image(panel)):
                        file.mux(packet)
                for packet in stream.encode():
                    file.mux(packet)
            hidden_indices = np.flatnonzero(redacted[in_range])
            slots = sorted(set([int(hidden_indices[0]), int(hidden_indices[len(hidden_indices) // 2]), int(hidden_indices[-1])]))
            contact = Image.new("RGB", (w * 6, (h + 24) * len(slots)))
            for i, slot in enumerate(slots):
                contact.paste(panels[slot], (0, i * (h + 24)))
            contact.save(output / "comparison.jpg", quality=94)
            with av.open(str(output / "comparison.mp4")) as file:
                assert sum(1 for _ in file.decode(video=0)) == len(panels)
            report = dict(case=case_name, metrics=metrics, native_indices=np.flatnonzero(in_range).tolist(),
                          fixed_hidden_count=int(hidden[in_range].sum()),
                          reference="ViTPose on actual unoccluded native RGB; not human labels or 3D ground truth",
                          unavailable_model_slots=[59, 60], quality_passed=False,
                          visual_scope="Automatic full decode only; manual review separately recorded",
                          protocol_sha256=hashlib.sha256((OUT / "RETARGET_PROTOCOL.json").read_bytes()).hexdigest(),
                          media_frames=len(panels), media_columns=columns,
                          media_colors="green prediction, red hidden reference detection, yellow visible reference detection")
            (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            links.append(f'<h2>{case_name}</h2><video controls preload="metadata" width="100%" src="cases/{case_name}/evaluation/comparison.mp4"></video><p><a href="cases/{case_name}/evaluation/comparison.jpg">遮挡窗口对照图</a></p>')
    (OUT / "metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    with (OUT / "metrics.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "gates.json").write_text(json.dumps(verdicts, indent=2, allow_nan=False) + "\n")
    (OUT / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>文字辅助遮挡动作补全</title><body style="max-width:1600px;margin:auto;background:#eee"><h1>CondMDI 文字辅助动作补全</h1><p>真实原始观测｜遮挡输入｜O2｜RoHM｜CondMDI 无文字｜CondMDI 有文字。均为相机投影诊断，并非新生成的 4DAnyone RGB；隐藏原图仅供评测。红点为隐藏参考检测，绿线为模型投影，黄点为可见参考检测。</p><p><a href="metrics.csv">指标 CSV</a> · <a href="gates.json">逐条保护线</a></p>' + "".join(links) + "</body>")
    (OUT / "evaluation_status.json").write_text(json.dumps(dict(state="finished_needs_visual_review", cases=protocol["cases"], quality_approved=False), indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
