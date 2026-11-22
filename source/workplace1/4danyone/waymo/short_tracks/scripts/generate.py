"""Explicit experiment adapter: reuse verified motion and retain conditioning."""
import argparse
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

BASE=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('FDANYONE_WAYMO_ROOT',str(BASE))).resolve()
REPO=BASE/'short_tracks/repo'
sys.path.insert(0,str(REPO))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','turbo']);args=parser.parse_args()
    review=json.loads((ROOT/'preflight/review.json').read_text())
    assert review['allow_generation'], 'Motion and mask review required'
    from fdanyone import pipeline
    from fdanyone.motion.result import MotionResult
    from fdanyone.skeleton.pipeline import Conditioning
    from inference import inference
    original_conditioning=pipeline._build_conditioning
    def motion_from_preflight(**kwargs):
        # The pipeline subsequently validates input byte size, mtime, source PTS and geometry.
        return MotionResult.load(ROOT/'preflight/motion')
    def retained_conditioning(**kwargs):
        fixed_views=int(os.environ.get('FDANYONE_RETAINED_VIEWS','0'))
        if fixed_views:
            assert fixed_views==(6 if args.mode=='smoke' else 24)
        if args.mode=='smoke' or fixed_views:
            return Conditioning.load(ROOT/'preflight/conditioning')
        kwargs['output_dir']=ROOT/'conditioning24'
        return original_conditioning(**kwargs)
    pipeline._run_motion=motion_from_preflight
    pipeline._build_conditioning=retained_conditioning
    out=ROOT/'outputs'/args.mode
    if out.exists():raise FileExistsError(out)
    status={'scope':'waymo_public_weight_test','training':False,'stage':args.mode,'state':'running',
            'started_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'motion':'preflight/motion',
            'adapter':'verified motion reuse and retained conditioning; official DiT unchanged'}
    (ROOT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    try:
        inference(video_path=str(ROOT/'inputs/main.mp4'),output_dir=str(out),views_per_layer=6 if args.mode=='smoke' else 24,
            gpu_ids=[0] if args.mode=='smoke' else [0,1,2,3],attention_backend='sdpa',enable_turbo=True,target_fps=25,seed=42,
            model_dir=str(REPO/'models'),gvhmr_root=str(REPO/'third_party/GVHMR'))
        status['state']='generated_validation_pending'
    except Exception as error:
        status['state']='failed';status['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        status['updated_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
        (ROOT/'status.json').write_text(json.dumps(status,indent=2)+'\n')


if __name__=='__main__':main()
