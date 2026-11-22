"""R07 target, original5k, ordinary15k, sigma8-15k; same cameras and times."""
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
from evaluate_asset61 import Renderer,sha,save
from render61_helpers import pane,uint8
from video_io import MP4Writer

R=Path(__file__).resolve().parents[1];W=R.parent


def main():
    
    label='r07_short61_24';gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);assert gpu in (4,5,6,7)
    expected=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-')
    torch.cuda.init();actual=str(torch.cuda.get_device_properties(0).uuid);assert actual==expected and torch.cuda.device_count()==1
    root=R/'diagnostics/weak_velocity'
    assert json.loads((root/'paired_audit_sigma8.json').read_text())['sigma_only_parameter_changes']['zero']==['durations']
    paths=[('Original5k / sigma15',R/'reconstruction/outputs/r07_short61_24/ckpts/ckpt_4999.pt'),
           ('Ordinary15k / sigma15',root/'zero/ckpts/ckpt_14999.pt'),
           ('Optimized15k / sigma8',root/'zero_sigma8/ckpts/ckpt_14999.pt')]
    data=R/'reconstruction/data'/label;models=[Renderer(path,data) for _,path in paths]
    if any(not np.array_equal(model.parser.transform,models[0].parser.transform) for model in models):raise ValueError('Model coordinate frames differ')
    output=root/'before_after';output.mkdir(parents=True,exist_ok=False)
    width=352*(1+len(paths))
    views=[0,8,16];writers={};counts=[]
    with ExitStack() as stack:
        for view in views:
            writer=MP4Writer(str(output/f'view{view:02d}.mp4'),25,width,664);writers[view]=writer;stack.callback(writer.close)
        for frame in range(61):
            for view in views:
                with Image.open(data/'images'/f'cam{view:02d}'/f'{frame:06d}.png') as im:rgba=np.asarray(im.convert('RGBA'),dtype=np.float32)/255
                target=rgba[...,:3]*rgba[...,3:]+(1-rgba[...,3:]);panels=[pane(Image.fromarray(uint8(target)),'Generated target + fixed mask')]
                for (name,_),model in zip(paths,models,strict=True):
                    rgb,alpha=model.render(view,frame);count=int((alpha>.1).sum())
                    if count<100:raise ValueError('Empty model at comparison camera/time')
                    counts.append(count);image=np.clip(rgb+(1-alpha[...,None]),0,1)
                    panels.append(pane(Image.fromarray(uint8(image)),name))
                array=np.concatenate(panels,axis=1);writers[view].append_data(array)
                if frame in (0,10,20,30,40,50,60):Image.fromarray(array).save(output/f'view{view:02d}_frame{frame:03d}.jpg',quality=95)
    videos={}
    for view in views:
        path=output/f'view{view:02d}.mp4'
        with imageio.get_reader(str(path),input_params=['-threads','1']) as reader:
            meta=reader.get_meta_data();count=sum(1 for _ in reader)
            assert count==61 and meta['fps']==25 and tuple(meta['size'])==(width,664)
        videos[path.name]={'frames':count,'fps':25,'sha256':sha(path)}
    save(output/'validation.json',{'state':'full_time_budget_comparison_rendered_decoded_visual_review_pending','gpu':gpu,'cuda_uuid':actual,
        'videos':videos,'models':[{'name':name,'checkpoint':str(path),'sha256':sha(path)} for name,path in paths],
        'output_camera_resolution':[352,640],'views':views,'model_times':'frame/60','playback_seconds':'frame/25',
        'minimum_foreground_pixels':min(counts),'filtering':False,
        'mode':'baseline_budget_then_initial_duration',
        'scope':'Same61-frame data, camera/time and11222 points. Two15k runs differ only initial duration. Old5k uses a different training and learning-rate schedule; not a duration-only comparison. No filtering or enhancement; generated targets are not real novel-view GT.'})
    print(json.dumps({'label':label,'state':'rendered_and_decoded'}),flush=True)


if __name__=='__main__':main()
