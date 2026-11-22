"""Reusable prompted accessory tracking + original-RGB alpha composition.

Prompts identify visible carried objects, not arbitrary foreground expansion.
No automatic claim that a detected/segmented object belongs to the target.
"""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
import time
from gate import sha

PROJECT = Path('/mnt/ojc/workplace1/4danyone')


def composite(rgb, alpha):
    import numpy as np
    a = alpha.astype(np.float32)[..., None] / 255
    return np.rint(rgb * a + 255 * (1-a)).clip(0, 255).astype(np.uint8)


def merge_masks(body, objects):
    import numpy as np
    if body.dtype != np.uint8 or objects.dtype != np.uint8 or objects.ndim != 4:
        raise ValueError('Expected uint8 body [T,H,W] and objects [K,T,H,W]')
    if objects.shape[1:] != body.shape or not len(objects):
        raise ValueError('Object/body shape mismatch or empty object set')
    return np.maximum(body, objects.max(axis=0))


def run(plan_file, output):
    import av
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageOps
    plan = json.loads(plan_file.read_text())
    if plan['schema'] != 'accessory-repair-plan-v1':
        raise ValueError('Unknown repair plan')
    source = Path(plan['source_rgb']); mask_file = Path(plan['body_masks'])
    if sha(source) != plan['source_sha256'] or sha(mask_file) != plan['body_masks_sha256']:
        raise ValueError('Input changed after prompt definition')
    body = np.load(mask_file, allow_pickle=False)[plan['body_mask_key']]
    if body.dtype != np.uint8 or body.ndim != 3:
        raise ValueError('Body masks must be uint8 [T,H,W]')
    count, height, width = body.shape
    frame_dir = Path(plan['sam_frames'])
    jpgs = sorted(frame_dir.glob('*.jpg'))
    if len(jpgs) != count or [p.stem for p in jpgs] != [f'{i:05d}' for i in range(count)]:
        raise ValueError('SAM frames do not match mask timeline')
    if not plan['objects'] or len({r['id'] for r in plan['objects']}) != len(plan['objects']):
        raise ValueError('Need uniquely identified carried objects')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Run under one GPU lease')
    uuid = str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    if uuid != os.environ['FD_EXPECTED_GPU_UUID'].removeprefix('GPU-'):
        raise RuntimeError('GPU UUID mismatch')
    output.mkdir(parents=True, exist_ok=False)
    (output/'PLAN.json').write_bytes(plan_file.read_bytes())
    sys.path.insert(0, str(PROJECT/'waymo/segmentation'))
    from sam2.build_sam import build_sam2_video_predictor
    from sam_transfer_guard import install_transfer_guard
    started = time.time()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',
            str(PROJECT/'waymo/segmentation/sam2.1_hiera_large.pt'), device='cuda', apply_postprocessing=False)
        install_transfer_guard(predictor)
        state = predictor.init_state(video_path=str(frame_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
        ids = []
        for item in plan['objects']:
            ident = item['id']; ids.append(ident)
            if type(ident) is not int or ident <= 0:
                raise ValueError('Object IDs must be positive integers')
            if not item['ownership_basis'] or not any(a['frame'] == 0 for a in item['annotations']):
                raise ValueError('Need target ownership evidence and initial frame prompt')
            for annotation in item['annotations']:
                t = annotation['frame']
                pts = np.asarray(annotation['points_xy'], np.float32)
                labels = np.asarray(annotation['labels'], np.int32)
                box = np.asarray(annotation['box_xyxy'], np.float32)
                if not (0 <= t < count and pts.ndim == 2 and pts.shape[1] == 2 and len(pts) == len(labels)):
                    raise ValueError('Invalid prompt geometry')
                if not np.isfinite(pts).all() or not np.isfinite(box).all() or not np.isin(labels,[0,1]).all():
                    raise ValueError('Nonfinite prompt or invalid labels')
                if box.shape != (4,) or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                    raise ValueError('Box outside crop')
                if not ((pts >= 0).all() and (pts[:,0] < width).all() and (pts[:,1] < height).all()):
                    raise ValueError('Points outside crop')
                predictor.add_new_points_or_box(state, frame_idx=t, obj_id=ident, points=pts, labels=labels, box=box)
        objects = np.empty((len(ids), *body.shape), np.uint8); seen = []
        for t, actual_ids, logits in predictor.propagate_in_video(state):
            if list(actual_ids) != ids or not torch.isfinite(logits).all():
                raise RuntimeError('Invalid SAM object ordering/output')
            objects[:, t] = np.rint(logits[:,0].sigmoid().float().cpu().numpy()*255).astype(np.uint8)
            seen.append(t)
    if seen != list(range(count)):
        raise RuntimeError('Incomplete object propagation')
    combined = merge_masks(body, objects)
    np.savez_compressed(output/'masks.npz', objects=objects, combined=combined)
    picked = set(np.linspace(0,count-1,min(12,count),dtype=int).tolist())
    sheet = Image.new('RGB',(800, len(picked)*264),(128,128,128))
    video = output/'corrected_input.mp4'; temporary=output/'corrected_input.writing.mp4'
    clocks=[]; rates=[]; j=0
    with av.open(str(source)) as inp, av.open(str(temporary),'w') as out:
        inp.streams.video[0].thread_count=2
        rate=inp.streams.video[0].average_rate
        stream=out.add_stream('libx264rgb',rate=rate)
        stream.width=width;stream.height=height;stream.pix_fmt='rgb24'
        stream.options={'crf':'0','preset':'fast','threads':'2'}
        for i, frame in enumerate(inp.decode(video=0)):
            if i >= count: raise ValueError('Source has extra frames')
            rgb=frame.to_ndarray(format='rgb24')
            if rgb.shape != (height,width,3): raise ValueError('Source/mask geometry mismatch')
            clock=Fraction(frame.pts)*frame.time_base;clocks.append(clock)
            if clock != Fraction(i,1)/rate: raise ValueError('Noncanonical source clock; normalize explicitly first')
            corrected=composite(rgb,combined[i])
            vf=av.VideoFrame.from_ndarray(corrected,format='rgb24');vf.pts=i;vf.time_base=1/rate
            for packet in stream.encode(vf): out.mux(packet)
            if i in picked:
                overlay=rgb.copy();added=combined[i]>body[i]
                overlay[added]=(overlay[added]*.5+np.array([255,0,255])*.5).astype(np.uint8)
                for col,(arr,title) in enumerate([(rgb,'SOURCE'),(composite(rgb,body[i]),'BEFORE'),(corrected,'AFTER'),(overlay,'ADDED ALPHA')]):
                    sheet.paste(ImageOps.contain(Image.fromarray(arr),(200,240)),(col*200,j*264+24))
                    ImageDraw.Draw(sheet).text((col*200+3,j*264+3),f'{title} f{i}',fill='white')
                j+=1
        for packet in stream.encode(): out.mux(packet)
    if len(clocks)!=count: raise ValueError('Source truncated')
    decoded=0
    # Lossless RGB and exact timestamps checked against a fresh source decode.
    with av.open(str(temporary)) as got, av.open(str(source)) as src:
        for i,(g,s) in enumerate(zip(got.decode(video=0),src.decode(video=0),strict=True)):
            if Fraction(g.pts)*g.time_base != clocks[i]: raise ValueError('Output clock changed')
            if not np.array_equal(g.to_ndarray(format='rgb24'),composite(s.to_ndarray(format='rgb24'),combined[i])):
                raise ValueError('Output differs from source-RGB alpha composition')
            decoded+=1
    if decoded!=count: raise ValueError('Output truncated')
    temporary.replace(video);sheet.save(output/'input_contact.jpg',quality=94)
    area=(objects>127).sum(axis=(2,3))
    audit=dict(state='repaired_input_requires_review',plan_sha256=sha(plan_file),frames=count,fps=str(rate),
        source_sha256=sha(source),body_masks_sha256=sha(mask_file),input_sha256=sha(video),masks_sha256=sha(output/'masks.npz'),
        gpu_uuid=uuid,seconds=time.time()-started,body_alpha_decrease_pixels=int((combined<body).sum()),
        object_area_per_frame=area.tolist(),added_alpha_pixels_per_frame=(combined>body).sum((1,2)).tolist(),
        sam_frame_sha256=[sha(p) for p in jpgs],review_frames=sorted(picked),generation_allowed=False,
        warnings=['Prompts are per-person source annotations, not automatic accessory recognition.',
                  'Review drift/occlusion/bystanders; area statistics do not prove ownership.',
                  'Only visible source RGB retained; hidden object shape/color not guaranteed.'])
    (output/'AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.plan,a.output)
