"""Frozen, train-only uint8 anatomical-core cache for the authorized ablation."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from prepare_diagnostics import ROOT, BASE, conservative, project
from render_baseline import sha, save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args()
    assert args.label in ['N04','N11'];cv2.setNumThreads(2)
    sample=BASE/'samples'/args.label
    data=json.loads((BASE/'four_dg/data'/args.label/'fdanyone.json').read_text())
    keypoints=np.load(sample/'generation/preflight/conditioning/keypoints_3d.npy')
    cams=json.loads((sample/'generation/outputs/turbo/cameras.json').read_text())['cameras']
    out=ROOT/'optimization/data'/args.label;out.mkdir(parents=True,exist_ok=False)
    records=[]
    for view in data['train_views']:
        c=next(c for c in cams if c['camera_id']==view);k=np.asarray(c['K'],float);k[:2]*=.5
        path=out/f'view{view:02d}.npy'
        labels=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=(data['frame_count'],640,352))
        for frame in range(data['frame_count']):
            original=sample/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'
            mask=np.asarray(Image.open(original))[...,3]>127
            uv,_=project(keypoints[frame],c['camera_to_world'],k)
            labels[frame],_=conservative(uv,mask,args.label,view)
        labels.flush();del labels
        records.append(dict(view=view,path=str(path),sha256=sha(path)))
    save(out/'complete.json',dict(state='complete',label=args.label,train_views=data['train_views'],files=records,
        core_definition_sha256=sha(ROOT/'scripts/prepare_diagnostics.py'),heldout_content_read=False,
        scope='Conservative projected anatomical cores intersected with frozen generated reference foreground; unknown retains global RGB supervision'))


if __name__=='__main__':main()
