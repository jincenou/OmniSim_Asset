"""Validate all generated videos and preserve the retained root-local cameras."""
import json,hashlib,shutil
from pathlib import Path

def finish(base,frames):
 import av
 gen=base;source=gen/'outputs/turbo';cond=gen/'preflight/conditioning';meta=json.loads((source/'metadata.json').read_text());assert meta['output']['frames_per_video']==frames
 cameras=json.loads((source/'cameras.json').read_text());retained=json.loads((cond/'cameras.json').read_text());assert len(cameras['cameras'])==len(retained['cameras'])==24
 # The official pipeline saves the retained conditioning cameras verbatim.
 import numpy as np
 for actual,expected in zip(cameras['cameras'],retained['cameras'],strict=True):
  for k in ['K','camera_to_world']:assert np.allclose(actual[k],expected[k],atol=1e-8),f'Generated physical camera mismatch:{k}'
  for k in ['camera_id','image_width','image_height']:assert actual[k]==expected[k],k
 assert cameras['world_frame']['name']=='root_local_human_world'
 rows=[]
 for view in range(24):
  path=source/'videos'/f'{view:02d}.mp4'
  with av.open(str(path)) as con:
   con.streams.video[0].thread_count=1;count=0;sampled={}
   for i,f in enumerate(con.decode(video=0)):
    assert (f.width,f.height)==(704,1280) and abs(float(f.pts*f.time_base)-i/25)<1e-8;count+=1
    if i in [0,frames//2,frames-1]:sampled[i]=f.to_ndarray(format='rgb24').astype(np.float32)/255
  assert count==frames
  rows.append({'view':view,'frames':count,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'first_middle_RGB_RMSE':float(np.sqrt(np.square(sampled[0]-sampled[frames//2]).mean())),'first_last_RGB_RMSE':float(np.sqrt(np.square(sampled[0]-sampled[frames-1]).mean())),'temporal_diagnostic_scope':'Pixel change only;not proof of accurate motion or identity'})
 # cameras.json refers to ../root_motion.npz; copy the tiny map to that exact path.
 shutil.copy2(gen/'preflight/root_motion.npz',source/'root_motion.npz')
 shutil.copy2(gen/'research/root_local.json',source/'root_local.json')
 cameras['root_motion']={'file':'root_motion.npz','sha256':hashlib.sha256((source/'root_motion.npz').read_bytes()).hexdigest(),'local_to_canonical_key':'local_to_canonical','time_key':'frame_times','motion_world_to_local_key':'motion_to_local','provenance':'root_local.json'}
 (source/'cameras.json').write_text(json.dumps(cameras,indent=2)+'\n')
 (gen/'research/generated_validation.json').write_text(json.dumps({'state':'all24_videos_decoded_same_retained_cameras','videos':rows,'production_approved':False,'visual_review':'pending'},indent=2)+'\n')

if __name__=='__main__':
 import sys
 from prepare_person import selected,ROOT
 selected(sys.argv[1]);finish(ROOT/'frontend'/sys.argv[1],121)
