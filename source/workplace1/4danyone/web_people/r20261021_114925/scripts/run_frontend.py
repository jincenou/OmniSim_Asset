"""Bind the recorded G0 conditioning/generation adapters to this run only."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path('/mnt/ojc/workplace1/4danyone')
EVIDENCE=ROOT/'research/baseline_evidence'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['conditioning','generate']);parser.add_argument('id');args=parser.parse_args()
    sys.path.insert(0,str(ROOT/'scripts'))
    from prepare_person import selected,sha
    selected(args.id)
    if args.stage == 'generate':
        sys.path.insert(0, str(PROJECT/'preprocessing/accessory_guard'))
        from gate import check_root
        check_root(ROOT/'frontend'/args.id)
    lock=ROOT/'BASELINE_LOCK.json';assert sha(lock)==lock.with_suffix('.sha256').read_text().strip()
    spec=json.loads(lock.read_text())
    for row in spec['frontend']['geometry_evidence']:
        assert sha(ROOT/row['copy'])==row['sha256']
    # These historical adapters explicitly check a numeric physical index. Resolve
    # the lease UUID before importing torch; the assigned card remains unchanged.
    uuid=os.environ['FD_EXPECTED_GPU_UUID']
    index=subprocess.check_output(['nvidia-smi',f'--id={uuid}','--query-gpu=index','--format=csv,noheader'],text=True).strip()
    assert int(index) in (4,5,6,7)
    os.environ.update(CUDA_VISIBLE_DEVICES=index,FDANYONE_RUNTIME=str(PROJECT/'repo'),
        FDANYONE_FRAMES='121',FDANYONE_ROOT_VIEWS='24',FDANYONE_RETAINED_VIEWS='24',
        FDANYONE_WAYMO_ROOT=str(ROOT/'frontend'/args.id),
        FDANYONE_SOURCE_EXPERIMENT=str(ROOT/'inputs'/args.id),FDANYONE_MOTION_VARIANT='static_estimated')
    sys.path.insert(0,str(EVIDENCE))
    name='conditioning' if args.stage=='conditioning' else 'generate_single'
    module_spec=importlib.util.spec_from_file_location('_web_g0_'+name,EVIDENCE/f'{name}.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    if args.stage=='generate':
        gate=json.loads((ROOT/'inputs'/args.id/'research/motion_gate.json').read_text())
        assert gate['variants']['static_estimated']['allow_exploratory_generation']
        assert gate['visual_review']!='pending'
        # Preserve the actual RCP bank; no change to denoising, RNG or handoff.
        from fdanyone.model import inference as model_inference
        original_rcp=model_inference._denoise_rcp
        def retain_rcp(**kwargs):
            import torch
            value=original_rcp(**kwargs)
            path=ROOT/'frontend'/args.id/'preflight/rcp_latents.pt'
            assert not path.exists()
            torch.save(dict(latents=value,camera_ids=kwargs['camera_ids'],seed=kwargs['seed'],
                            scope='Exact generated proposal latents used by target generation; not final asset'),path)
            return value
        model_inference._denoise_rcp=retain_rcp
        sys.argv=[str(EVIDENCE/'generate_single.py'),'turbo']
    module.main()


if __name__=='__main__':main()
