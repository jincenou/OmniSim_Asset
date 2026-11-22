"""Create a pending, hash-bound review; never approve automatically."""
import argparse
import json
from pathlib import Path
from gate import sha, SCHEMA


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--mask',type=Path,required=True)
    p.add_argument('--frames',type=int,required=True)
    p.add_argument('--evidence',type=Path,nargs='+',required=True)
    a=p.parse_args();target=a.root/'preflight/accessory_review.json'
    if target.exists():raise FileExistsError(target)
    if a.frames<=0:raise ValueError('Positive frame count required')
    def record(path):return dict(path=str(path.resolve()),sha256=sha(path))
    files=dict(source_rgb=record(a.source),mask=record(a.mask),generation_input=record(a.root/'inputs/main.mp4'))
    meta=a.root/'preflight/conditioning/metadata.json'
    if meta.exists():files['conditioning_source']=record(meta.parent/json.loads(meta.read_text())['source_video'])
    review=dict(schema=SCHEMA,decision='pending',reviewer='',accessories='unknown',frames=a.frames,reviewed_frames=[],
        source_input_compared=False,no_new_bystander_leakage=False,no_visible_accessory_whiteout=False,
        occlusion_drift_checked=False,files=files,evidence=list(map(record,a.evidence)),
        notes='Review source vs input over the whole timeline. Record uncertain/occluded intervals; do not fill true without evidence.')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as out:json.dump(review,out,ensure_ascii=False,indent=2)
    print(target)

if __name__=='__main__':main()
