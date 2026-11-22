"""Deterministic dataset-person prompts and the existing foreground gate."""
from pathlib import Path
import os,json,sys,shutil
import numpy as np
RUN=Path(__file__).resolve().parents[1];W=RUN.parent;ROOT=Path(os.environ['FDANYONE_SAMPLE_ROOT']);COUNT=int(os.environ['FDANYONE_FRAMES'])
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 import av
 from PIL import Image
 mode=sys.argv[1]
 if mode=='prepare':
  (ROOT/'inputs').rename(ROOT/'inputs_rife');(ROOT/'inputs').mkdir();(ROOT/'segmentation/frames').mkdir(parents=True);(ROOT/'research').mkdir(exist_ok=True)
  mapping=json.loads((ROOT/'inputs_rife/frame_mapping.json').read_text());native=json.loads((ROOT/'native.json').read_text());candidate=json.loads((ROOT.parent/'candidate.json').read_text());save(ROOT/'candidate.json',candidate)
  mapping.update(scene=candidate['scene'],track_id=candidate['track_id'],camera_index=candidate['camera_index'],identity_key=candidate['identity_key'],source_split=candidate['split'],real_source_frames=len(native['frames']),output_frames=COUNT,real_source_frame_range=[candidate['observations'][0]['frame_index'],candidate['observations'][-1]['frame_index']])
  with av.open(str(ROOT/'inputs_rife/main.mp4')) as con:
   con.streams.video[0].thread_count=1
   for i,f in enumerate(con.decode(video=0)):f.to_image().save(ROOT/'segmentation/frames'/f'{i:05d}.jpg',quality=95,subsampling=0)
  assert i+1==COUNT
  for row in mapping['frames']:
   row['crop_xyxy']=[0,0,f.width,f.height];row['source_path']=native['frames'][row['left_native_index']]['raw_source_path']
  save(ROOT/'inputs_rife/frame_mapping.json',mapping);(ROOT/'inputs/cameras.npz').symlink_to(ROOT/'inputs_rife/cameras.npz')
 elif mode=='segment':
  import torch
  from gpu_guard import validate_gpu
  sys.path.insert(0,str(W/'segmentation'));from sam2.build_sam import build_sam2_video_predictor
  device=validate_gpu();mapping=json.loads((ROOT/'inputs_rife/frame_mapping.json').read_text());box=np.asarray(mapping['frames'][0]['bbox_input_xyxy'],np.float32);x0,y0,x1,y1=box;bw,bh=x1-x0,y1-y0
  points=np.asarray([[x0+.5*bw,y0+.15*bh],[x0+.5*bw,y0+.45*bh],[x0+.45*bw,y0+.8*bh]],np.float32);labels=np.ones(3,np.int32)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',str(W/'segmentation/sam2.1_hiera_large.pt'),device='cuda',apply_postprocessing=False)
   from sam_transfer_guard import install_transfer_guard
   install_transfer_guard(predictor)
   state=predictor.init_state(video_path=str(ROOT/'segmentation/frames'),offload_video_to_cpu=True,offload_state_to_cpu=True);predictor.add_new_points_or_box(state,frame_idx=0,obj_id=1,points=points,labels=labels,box=box)
   masks=[];seen=[]
   for index,ids,logits in predictor.propagate_in_video(state):
    assert ids==[1] and torch.isfinite(logits).all(), 'Invalid SAM logits';masks.append(logits[0,0].sigmoid().float().cpu().numpy().__mul__(255).round().astype(np.uint8));seen.append(index)
  assert seen==list(range(COUNT));np.savez_compressed(ROOT/'segmentation/target_masks.npz',masks=np.asarray(masks));save(ROOT/'segmentation/manifest.json',{'device':device,'points':points.tolist(),'labels':labels.tolist(),'box':box.tolist(),'frames':COUNT,'prompt':'fixed annotation-bbox-relative face/torso/leg points;no output-based manual reselection','visual_review':'pending','accessory_completeness':'pending'})
 elif mode=='composite':
  import hashlib
  sys.path.insert(0,str(W/'scripts'));from prepare import encode
  masks=np.load(ROOT/'segmentation/target_masks.npz')['masks'];mapping=json.loads((ROOT/'inputs_rife/frame_mapping.json').read_text());frames=[]
  with av.open(str(ROOT/'inputs_rife/main.mp4')) as con:
   con.streams.video[0].thread_count=1
   for i,f in enumerate(con.decode(video=0)):
    rgb=f.to_ndarray(format='rgb24');alpha=masks[i].astype(np.float32)[...,None]/255;frames.append(Image.fromarray(np.rint(rgb*alpha+255*(1-alpha)).clip(0,255).astype(np.uint8)))
  assert len(frames)==COUNT;encode(ROOT/'inputs/main.mp4',frames,25);mapping['video_sha256']=hashlib.sha256((ROOT/'inputs/main.mp4').read_bytes()).hexdigest();mapping['background_compositing']={'method':'straight alpha over white','mask_source':'segmentation/target_masks.npz','visual_review':'pending'};save(ROOT/'inputs/frame_mapping.json',mapping)
  from mask_screen import screen
  result=screen(ROOT,masks,mapping)
  if not result['preparation_usable']:raise RuntimeError('Preparation has no foreground in any frame; retain identity for other windows: '+str(result['reasons']))
 elif mode=='motion_gate':
  import hashlib
  motion=ROOT/'motion_pair/calibrated_moving';assert (motion/'motion.safetensors').is_file();save(ROOT/'research/motion_gate.json',{'input_sha256':hashlib.sha256((ROOT/'inputs/main.mp4').read_bytes()).hexdigest(),'kind':'automatic_numerical_preflight','visual_review':'pending','variants':{'calibrated_moving':{'allow_exploratory_generation':True,'motion_sha256':hashlib.sha256((motion/'motion.safetensors').read_bytes()).hexdigest()}},'limits':'Calibrated croppedK and exposure-aligned rotations used;camera translation/world scale and exact sourcepose not verified.'})
 else:raise ValueError(mode)
 print(json.dumps({'stage':mode,'state':'complete','sample':os.environ['FDANYONE_SAMPLE_ID']}),flush=True)
if __name__=='__main__':main()
