"""All paired context candidates: decoded video and diagnostics, never hidden GT."""
import json
from pathlib import Path
from types import SimpleNamespace
import av
import numpy as np
import torch
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from safetensors.torch import load_file
from review_motion import _gvhmr_geometry_context,REPO,EDGES
G=Path(__file__).resolve().parent
E=G/'text_motion/bilateral_context/repair_v1'

def main():
    args=SimpleNamespace(sample='P18_context')
    torch.set_num_threads(2)
    source=E/'source';result=E/'cases/P18'
    output=E/'evaluation';output.mkdir(exist_ok=False)
    raw=load_file(str(source/'motion/motion.safetensors'))
    template=load_file(str(result/'text/old_anchor/motion_candidates.safetensors'))
    reference_report=json.loads((result/'text/old_anchor/report.json').read_text())
    params={k:raw['smpl_params_incam.'+k] for k in ['body_pose','betas','global_orient','transl']}
    params['betas']=template['betas'];variants={'raw_context':params}
    for branch in ['no_text','text']:
        for arm in ['old_anchor','gap_free']:
            p=load_file(str(result/branch/arm/'motion_candidates.safetensors'))
            np.testing.assert_array_equal(p['times_seconds'],template['times_seconds'])
            np.testing.assert_array_equal(p['evidence_supported_visible_candidate'],template['evidence_supported_visible_candidate'])
            variants[branch+'_'+arm]=dict(params,body_pose=p['body_pose'],global_orient=p['incam.global_orient'],transl=p['incam.transl'])
    kp, K = raw["observed_keypoints_2d"].numpy(), raw["K_fullimg"].numpy()
    visible = template["evidence_supported_visible_candidate"].numpy()
    frames_meta = json.loads((source / "input/frame_mapping.json").read_text())["frames"]
    height = np.array([x["bbox_input_xyxy"][3] - x["bbox_input_xyxy"][1] for x in frames_meta])
    t = template["times_seconds"].numpy()
    dt, scale = np.diff(t), reference_report["body_scale_m"]
    projections, locals_, metrics = {}, {}, {}
    with _gvhmr_geometry_context(REPO / "third_party/GVHMR"):
        from hmr4d.utils.smplx_utils import make_smplx
        body = make_smplx("supermotion_coco17").eval()
        for name, p in variants.items():
            np.testing.assert_array_equal(p["betas"].numpy(), template["betas"].numpy())
            with torch.inference_mode():
                points = body(**p).numpy()
            assert np.isfinite(points).all() and (points[..., 2] > 0).all()
            proj = np.einsum("tij,tnj->tni", K, points)
            uv = proj[..., :2] / proj[..., 2:]
            projections[name] = uv
            nme = np.linalg.norm(uv - kp[..., :2], axis=-1) / height[:, None]
            rotation = Rotation.from_rotvec(p["global_orient"].numpy()).as_matrix()
            local = np.einsum("tji,tnj->tni", rotation, points - p["transl"].numpy()[:, None])
            local -= local[:, 11:13].mean(1, keepdims=True)
            locals_[name] = local
            velocity = np.diff(local, axis=0) / dt[:, None, None]
            accel = np.diff(velocity, axis=0) / ((dt[1:] + dt[:-1]) / 2)[:, None, None]
            # Geometric joint angle of the 3D estimated thigh/shin, not axis-angle magnitude.
            knee = []
            for hip, kn, ankle in [(11, 13, 15), (12, 14, 16)]:
                a, b = local[:, hip] - local[:, kn], local[:, ankle] - local[:, kn]
                cosine = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
                knee.append(180 - np.degrees(np.arccos(np.clip(cosine, -1, 1))))
            metrics[name] = dict(visible_count=int(visible.sum()),
                                 supported_detector_nme_mean=float(nme[visible].mean()),
                                 supported_detector_nme_p95=float(np.percentile(nme[visible], 95)),
                                 local_range=float(np.sqrt(np.mean(np.ptp(local, axis=0) ** 2)) / scale),
                                 local_acceleration=float(np.sqrt(np.mean(accel ** 2)) / scale),
                                 knee_flexion_geometric_degrees=np.stack(knee, axis=1).tolist())
    labels = ["observed input", *variants]
    with av.open(str(source / "input/model_input.mp4")) as file:
        file.streams.video[0].thread_count = 1
        frames = [frame.to_image() for frame in file.decode(video=0)]
    assert len(frames) == len(t)
    width = 256
    img_h = round(frames[0].height * width / frames[0].width / 2) * 2
    row_h = img_h + 26 + 222
    panels = []
    display_scale = min(110 / scale, 110)
    for i, frame in enumerate(frames):
        panel = Image.new("RGB", (width * len(labels), row_h), "#222222")
        draw = ImageDraw.Draw(panel)
        for ci, name in enumerate(labels):
            im = frame.copy()
            overlay = ImageDraw.Draw(im)
            if ci:
                for a, b in EDGES:
                    overlay.line([tuple(projections[name][i, a]), tuple(projections[name][i, b])], fill="lime", width=2)
                for j, (x, y, _) in enumerate(kp[i]):
                    overlay.ellipse((x - 2, y - 2, x + 2, y + 2), fill="yellow" if visible[i, j] else "orange")
            panel.paste(im.resize((width, img_h)), (ci * width, 26))
            draw.text((ci * width + 3, 5), f"{name} {i}/25s", fill="white")
            if ci:
                local = locals_[name][i]
                # Two orthographic views of root-local articulation, shared display scale.
                for vi, axis in enumerate([0, 2]):
                    center = np.array([ci * width + width * (.25 + .5 * vi), 26 + img_h + 105])
                    xy = local[:, [axis, 1]] * np.array([display_scale, -display_scale]) + center
                    for a, b in EDGES:
                        draw.line([tuple(xy[a]), tuple(xy[b])], fill="#66d9ef", width=2)
                    draw.text((ci * width + vi * width // 2 + 5, 26 + img_h + 200), "local front" if vi == 0 else "local side", fill="white")
            else:
                draw.text((5, 26 + img_h + 30), "below: estimated 3D", fill="white")
                draw.text((5, 26 + img_h + 50), "root motion removed", fill="white")
                draw.text((5, 26 + img_h + 70), "no hidden ground truth", fill="white")
        panels.append(panel)
    with av.open(str(output / "comparison.mp4"), "w") as file:
        stream = file.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = panels[0].width, row_h, "yuv420p"
        stream.options = {"crf": "19", "threads": "2"}
        for panel in panels:
            for packet in stream.encode(av.VideoFrame.from_image(panel)):
                file.mux(packet)
        for packet in stream.encode():
            file.mux(packet)
    slots = np.rint(np.linspace(0, len(frames) - 1, 5)).astype(int)
    contact = Image.new("RGB", (panels[0].width, row_h * len(slots)))
    for i, slot in enumerate(slots):
        contact.paste(panels[slot], (0, i * row_h))
    contact.save(output / "comparison.jpg", quality=94)
    with av.open(str(output / "comparison.mp4")) as file:
        assert sum(1 for _ in file.decode(video=0)) == len(frames)
    report = dict(sample=args.sample, metrics=metrics, decoded_frames=len(frames),
                  contact_slots=slots.tolist(), columns=labels, quality_approved=False,
                  real_hidden_motion_accuracy=None,
                  limits="Visible detector fit is not independent GT. Knee flexion is 3D-estimate geometry, not a target or knee cap. Root-local display removes root for inspection only; export parameters retain root and time. Human dynamic review remains separate.")
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(sample=args.sample, state="review_ready", quality_approved=False)))


if __name__ == "__main__":
    main()
