"""Synchronized, identical-camera full-time comparison of frozen budgets."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import json
import subprocess
import numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as imageio
import torch
from evaluate_asset import Renderer,sha,save
from replay_walking import pane,uint8
from video_io import MP4Writer

R=Path(__file__).resolve().parents[1];W=R.parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',choices=['phone_full','pilot_043_twentyfour','batch_0073_twentyfour','batch_0086_twentyfour','walk063_rootlocal24','batch_0084_twentyfour'],required=True)
    parser.add_argument('--mode',choices=['budget','optimization','validation'],default='budget')
    args=parser.parse_args();label=args.label;gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu in (4,5,6,7)
    expected=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-')
    torch.cuda.init();actual=str(torch.cuda.get_device_properties(0).uuid);assert actual==expected and torch.cuda.device_count()==1
    if label not in ('walk063_rootlocal24','batch_0084_twentyfour') or args.mode!='validation':
        raise ValueError('Temporal comparisons require the two frozen movement labels')
    paths=[('Dense baseline 50k',R/'quality'/f'{label}_capacity50k/ckpts/ckpt_49999.pt'),
           ('Relocation through45k',R/'quality'/f'{label}_dense_relocate45k/ckpts/ckpt_49999.pt'),
           ('Duration sigma <=8frames',R/'quality'/f'{label}_dense_duration8/ckpts/ckpt_49999.pt')]
    data=(R if label=='walk063_rootlocal24' else W)/'reconstruction/data'/label;models=[Renderer(path,data) for _,path in paths]
    if any(not np.array_equal(model.parser.transform,models[0].parser.transform) for model in models):raise ValueError('Model coordinate frames differ')
    output=R/'comparisons'/(label+'_temporal');output.mkdir(parents=True,exist_ok=False)
    width=352*(1+len(paths))
    views=[0,8,16];writers={};counts=[]
    with ExitStack() as stack:
        for view in views:
            writer=MP4Writer(str(output/f'view{view:02d}.mp4'),25,width,664);writers[view]=writer;stack.callback(writer.close)
        for frame in range(121):
            for view in views:
                with Image.open(data/'images'/f'cam{view:02d}'/f'{frame:06d}.png') as im:rgba=np.asarray(im.convert('RGBA'),dtype=np.float32)/255
                target=rgba[...,:3]*rgba[...,3:]+(1-rgba[...,3:]);panels=[pane(Image.fromarray(uint8(target)),'Generated target + fixed mask')]
                for (name,_),model in zip(paths,models,strict=True):
                    rgb,alpha=model.render(view,frame);count=int((alpha>.1).sum())
                    if count<100:raise ValueError('Empty model at comparison camera/time')
                    counts.append(count);image=np.clip(rgb+(1-alpha[...,None]),0,1)
                    panels.append(pane(Image.fromarray(uint8(image)),name))
                array=np.concatenate(panels,axis=1);writers[view].append_data(array)
                if frame in (0,60,120):Image.fromarray(array).save(output/f'view{view:02d}_frame{frame:03d}.jpg',quality=95)
    videos={}
    for view in views:
        path=output/f'view{view:02d}.mp4'
        with imageio.get_reader(str(path),input_params=['-threads','1']) as reader:
            meta=reader.get_meta_data();count=sum(1 for _ in reader)
            assert count==121 and meta['fps']==25 and tuple(meta['size'])==(width,664)
        videos[path.name]={'frames':count,'fps':25,'sha256':sha(path)}
    save(output/'validation.json',{'state':'full_time_budget_comparison_rendered_decoded_visual_review_pending','gpu':gpu,'cuda_uuid':actual,
        'videos':videos,'models':[{'name':name,'checkpoint':str(path),'sha256':sha(path)} for name,path in paths],
        'output_camera_resolution':[352,640],'views':views,'model_times':'frame/120','playback_seconds':'frame/25',
        'minimum_foreground_pixels':min(counts),'filtering':False,
        'mode':'temporal_controls',
        'scope':'Identical352x640 camera and time for every model. No filtering or enhancement. Only generated target, no real novel-view truth. All models use the same dense initialization positions/colors and point count. Duration control changes the temporal support bound and initial durations; relocation control changes its stop iteration. Single seed, no visual approval.' if args.mode!='budget' else 'Same352x640 camera and time for every model. Q2 was trained native resolution but rendered directly to this common preview grid, not enlarged or enhanced. Native-grid quantitative comparisons stored separately in native_results.json. Only generated target, no real novel-view truth.'})
    print(json.dumps({'label':label,'state':'rendered_and_decoded'}),flush=True)


if __name__=='__main__':main()
