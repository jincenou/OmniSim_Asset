"""Standalone dynamic replay from exported model/camera files; forbid source data reads."""
import argparse
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from grid_common import ROOT,BASE,load_asset,camera,raw_render,sha,save,verify


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--iteration',type=int,required=True)
    parser.add_argument('--view',type=int,default=0);parser.add_argument('--output',required=True);parser.add_argument('--background',type=float,choices=[0.,.5,1.],default=.5)
    args=parser.parse_args();device=verify();run=Path(args.run).resolve();out=Path(args.output).resolve()
    assert run.is_relative_to(ROOT) and out.is_relative_to(ROOT) and 0<=args.view<24
    assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    def guard(event,values):
        if event=='open' and isinstance(values[0],(str,bytes)):
            p=Path(os.fsdecode(values[0])).absolute()
            assert not p.is_relative_to(BASE),'Standalone replay may not read old source data or code'
    sys.addaudithook(guard)
    snapshot=json.loads((run/'snapshots'/f'{args.iteration}.json').read_text())
    for path,digest in snapshot['files'].items():assert sha(run/path)==digest
    model,deform,pipe,config=load_asset(run,args.iteration)
    data=json.loads((run/'asset_camera_data.json').read_text());w,h=data['width'],data['height']
    writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*'mp4v'),25.,(w,h));assert writer.isOpened()
    ref_path=run/'snapshots'/f'{args.iteration}_reference.npz';references=np.load(ref_path);errors=[]
    for key in references.files:
        v,t=key.split('_');rgb,alpha=raw_render(model,deform,pipe,camera(data,int(v[1:]),float(t[1:])))
        err=float(np.max(np.abs(torch.cat([rgb,alpha]).cpu().numpy()-references[key])));assert err<1e-5;errors.append(err)
    try:
        for frame in range(data['frame_count']):
            rgb,alpha=raw_render(model,deform,pipe,camera(data,args.view,frame));image=(rgb+args.background*(1-alpha)).clamp(0,1)
            pixels=np.rint(image.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            writer.write(cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR))
    finally:writer.release()
    cap=cv2.VideoCapture(str(out));count=0
    while True:
        ok,image=cap.read()
        if not ok:break
        assert image.shape==(h,w,3);count+=1
    cap.release();assert count==data['frame_count']
    save(out.with_suffix('.json'),dict(state='passed',run=str(run),iteration=args.iteration,view=args.view,frames=count,fps=25,background=args.background,device=device,
        independent_reference_max_error=max(errors),source_data_reads=0,source_data_protection='Python open audit rejects every path under historical BASE',asset_camera_data_sha256=sha(run/'asset_camera_data.json'),video_sha256=sha(out),
        limitations='Known-camera replay with native dynamic time; not new-action driving or world-scale registration'))


if __name__=='__main__':main()
