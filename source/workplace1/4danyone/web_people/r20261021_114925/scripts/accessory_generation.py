"""Single same-seed appearance-only ablation; preserves baseline motion/cameras."""
import argparse
from dataclasses import replace
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from prepare_person import ROOT, PROJECT, sha, save

PERSON='W02'
TRIAL=ROOT/f'diagnostics/accessory_{PERSON}_v1'
BASE=ROOT/'frontend'/PERSON
DEST=TRIAL/'frontend'/PERSON


def prepare():
    review=json.loads((TRIAL/'VISUAL_REVIEW.json').read_text())
    assert review['allow_same_seed_generation'], 'Input repair must be inspected before generation'
    sys.path.insert(0,str(PROJECT/'repo'))
    from fdanyone.video import decode_canonical_clip, write_lossless_video, verify_lossless_video
    from fdanyone.motion.result import MotionResult
    import torch
    old=decode_canonical_clip(str(ROOT/'inputs'/PERSON/'source_crop.mp4'),num_frames=121,fps=Fraction(25))
    new=decode_canonical_clip(str(TRIAL/'corrected_input.mp4'),num_frames=121,fps=Fraction(25))
    assert old.width==new.width and old.height==new.height
    assert [(x.source_index,x.source_pts,x.canonical_timestamp) for x in old.frames]==[(x.source_index,x.source_pts,x.canonical_timestamp) for x in new.frames]
    motion=MotionResult.load(BASE/'preflight/motion');motion.validate_against_clip(old)
    # The actual GVHMR was computed from original full-frame RGB, not either
    # masked appearance clip. Rebind ONLY the downstream appearance-file identity.
    revised=replace(motion,source_size_bytes=new.source_size_bytes,source_mtime_ns=new.source_mtime_ns)
    revised.validate_against_clip(new)
    DEST.mkdir(parents=True,exist_ok=False)
    (DEST/'inputs').mkdir();(DEST/'preflight').mkdir();(DEST/'research').mkdir()
    (DEST/'inputs/main.mp4').symlink_to(TRIAL/'corrected_input.mp4')
    revised.save(DEST/'preflight/motion')
    assert sha(DEST/'preflight/motion/motion.safetensors')==sha(BASE/'preflight/motion/motion.safetensors')
    cond=DEST/'preflight/conditioning';cond.mkdir()
    meta=json.loads((BASE/'preflight/conditioning/metadata.json').read_text())
    for path in (BASE/'preflight/conditioning').iterdir():
        if path.name not in ('metadata.json',meta['source_video']):
            (cond/path.name).symlink_to(path)
    source_name=meta['source_video']
    write_lossless_video(new,cond/source_name);verify_lossless_video(new,cond/source_name)
    meta['appearance_only_ablation']=dict(source='corrected person+bag alpha from original RGB',
        framing='Frozen baseline crop/cameras; no framing or skeleton recalculation',
        motion='Original full-frame GVHMR tensors unchanged; masked-file byte identity rebound with explicit paired timeline check')
    save(cond/'metadata.json',meta)
    shutil.copy2(BASE/'preflight/root_motion.npz',DEST/'preflight/root_motion.npz')
    shutil.copy2(BASE/'research/root_local.json',DEST/'research/root_local.json')
    save(DEST/'preflight/review.json',dict(allow_generation=True,scope='User-authorized isolated accessory repair ablation',input_review=str(TRIAL/'VISUAL_REVIEW.json')))
    save(TRIAL/'GENERATION_PLAN.json',dict(seed=42,turbo_steps=4,views=24,frames=121,
        original_input_sha256=sha(ROOT/'inputs'/PERSON/'source_crop.mp4'),corrected_input_sha256=sha(TRIAL/'corrected_input.mp4'),
        motion_tensors_sha256=sha(DEST/'preflight/motion/motion.safetensors'),
        camera_sha256=sha(cond/'cameras.json'),baseline_camera_sha256=sha(BASE/'preflight/conditioning/cameras.json'),
        modifications=['Appearance input alpha union only','Downstream clip-file identity rebind with preserved motion tensors/timeline'],
        baseline_preserved=True,success='Not established until regenerated views are inspected; no backend recipe change'))
    print('Separate same-seed appearance branch prepared; motion tensor bytes and cameras unchanged')


def generate():
    assert json.loads((TRIAL/'VISUAL_REVIEW.json').read_text())['allow_same_seed_generation']
    plan=json.loads((TRIAL/'GENERATION_PLAN.json').read_text())
    assert sha(TRIAL/'corrected_input.mp4')==plan['corrected_input_sha256']
    uuid=os.environ['FD_EXPECTED_GPU_UUID']
    index=subprocess.check_output(['nvidia-smi',f'--id={uuid}','--query-gpu=index','--format=csv,noheader'],text=True).strip()
    assert int(index) in (4,5,6,7)
    os.environ.update(CUDA_VISIBLE_DEVICES=index,FDANYONE_RUNTIME=str(PROJECT/'repo'),
        FDANYONE_WAYMO_ROOT=str(DEST),FDANYONE_RETAINED_VIEWS='24')
    source=ROOT/'research/baseline_evidence/generate_single.py'
    spec=importlib.util.spec_from_file_location('_accessory_generate',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from fdanyone.model import inference
    original=inference._denoise_rcp
    def retain(**kwargs):
        import torch
        value=original(**kwargs)
        torch.save(dict(latents=value,camera_ids=kwargs['camera_ids'],seed=kwargs['seed']),DEST/'preflight/rcp_latents.pt')
        return value
    inference._denoise_rcp=retain
    sys.argv=[str(source),'turbo'];module.main()
    from finish_generation import finish
    finish(DEST,121)
    import publish_previews
    publish_previews.ROOT=TRIAL
    publish_previews.generated(PERSON)
    print('Corrected 24-view generation complete; visual comparison still required')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','generate'])
    parser.add_argument('--person',choices=['W02','W04'],default='W02')
    args=parser.parse_args();PERSON=args.person
    TRIAL=ROOT/f'diagnostics/accessory_{PERSON}_v1';BASE=ROOT/'frontend'/PERSON;DEST=TRIAL/'frontend'/PERSON
    {'prepare':prepare,'generate':generate}[args.stage]()
