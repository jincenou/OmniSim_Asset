"""Replay a native dynamic asset without reading target images or old experiment code."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--asset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['motion','orbit'],default='motion');p.add_argument('--view',type=int,default=0);p.add_argument('--time-sec',type=float,default=1.2);a=p.parse_args()
import torch
from runtime import load,render,device_record
from data import sha,save_json,model_time
from video_tools import Video,panel,uint8,camera_orbit
record=device_record();asset=json.loads((a.asset/'asset.json').read_text());assert sha(a.asset/'checkpoint.pt')==asset['checkpoint_sha256'];model,_=load(a.asset/'checkpoint.pt');manifest=json.loads((a.asset/'manifest.json').read_text());cams={c['id']:c for c in manifest['cameras']};writer=Video(a.output,352,680,fps=manifest.get('render_fps',25))
with torch.no_grad():
 if a.mode=='motion':
  for sec in manifest['render_timestamps_sec']:
   out=render(model,cams[a.view],model_time(manifest,sec),bg=1.);writer.add(panel(uint8(out['render'].permute(1,2,0).cpu().numpy()),f'Native unfiltered 4D asset / view{a.view}\nAction time {sec:.2f}s'))
 else:
  assert manifest['render_timestamps_sec'][0]<=a.time_sec<=manifest['render_timestamps_sec'][-1]
  for c in camera_orbit(manifest['cameras']):
   out=render(model,c,model_time(manifest,a.time_sec),bg=1.);writer.add(panel(uint8(out['render'].permute(1,2,0).cpu().numpy()),f'Geometry orbit / action FROZEN {a.time_sec:.2f}s'))
video=writer.close();save_json(a.output.with_suffix('.json'),{'state':'standalone_replay_complete','device':record,'video':video,'target_images_read':False,'old_code_used':False})
