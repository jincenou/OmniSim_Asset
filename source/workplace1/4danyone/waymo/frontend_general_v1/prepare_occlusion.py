"""Freeze and build real RGB occlusion controls before feature extraction."""
from pathlib import Path
import json,hashlib,datetime,shutil
import numpy as np
from PIL import Image,ImageDraw
G=Path(__file__).resolve().parent;D=G.parent/'frontend_quality_20261020';O=G/'occlusion_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def rectangle(box,w,h):
 x0,y0,x1,y1=box;bw=x1-x0;bh=y1-y0
 return [max(0,int(np.floor(x0-.1*bw))),max(0,int(np.floor(y0+.45*bh))),min(w,int(np.ceil(x1+.1*bw))),min(h,int(np.ceil(y1+.05*bh)))]
def main():
 O.mkdir(exist_ok=False)
 protocol={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'engineering_choices_not_paper_defaults':True,
 'subjects':['P24','P36'],'source_review':{'P24':'walking; native frames6..18 reviewed; lower limbs visible with normal self-occlusion; pole adjacent','P36':'visually standing; native frames6..18 reviewed; metric world-static truth unavailable'},
 'durations_seconds':[.2,.5,1.0],'center_seconds':1.2,'window':'native actual relative exposure t >= center-duration/2 and t < center+duration/2; record exact selected times',
 'occluder':'RGB128 opaque rectangle x=[bbox_x0-.1w,bbox_x1+.1w],y=[bbox_y0+.45h,bbox_y1+.05h],clipped; deliberately synthetic, not vehicle realism',
 'branches':{'rgb':'redact native RGB before RIFE; extract all image features and raw detector points anew; suppress known-occluded detections after extraction','kp_only':'reuse original image features and keypoints; suppress known-occluded keypoint coordinates and confidence; explicitly has unoccluded visual evidence'},
 'known_occlusion_support':'union of redacted endpoint rectangles on interpolated slots, dilated3pixels; both endpoints redacted defines core evaluation time; boundaries reported separately',
 'hidden_evaluation':'original native RGB detections extracted only in independent evaluator, not motion input; 2D proxy, not GT3D; avoid treating original GVHMR as truth',
 'optimization':'O1 and existing O2 local_temporal weights/budget unchanged; trusted evidence uses known target bbox support excluding artificial occluder, no hidden original instance mask',
 'fairness':'same measured camera, target identity boxes, checkpoint and time window; use known synthetic occlusion support only, not a validated real occlusion detector',
 'acceptance_protocol_sha256':sha(G/'protocol_v1.json'),'O2_protocol_sha256':sha(G/'protocol_v2.json'),'quality_passed':False}
 save(O/'PROTOCOL.json',protocol);(O/'PROTOCOL.sha256').write_text(sha(O/'PROTOCOL.json')+'\n')
 jobs=[]
 for sample in protocol['subjects']:
  src=D/'motion_unmasked'/sample;native=json.loads((src/'native.json').read_text());times=np.load(src/'native_cameras.npz')['image_center_timestamps_s'];times=times-times[0]
  mapping=json.loads((src/'inputs/frame_mapping.json').read_text())['frames'];w,h=native['size']
  for duration in protocol['durations_seconds']:
   case=O/f'{sample}_{round(duration*1000):04d}ms';case.mkdir();redacted=(times>=1.2-duration/2)&(times<1.2+duration/2)
   rects=np.array([rectangle(r['bbox_input_xyxy'],w,h) for r in native['frames']]);visible_mask=np.zeros((len(mapping),h,w),np.uint8);occluders=np.zeros_like(visible_mask);core=[]
   for i,row in enumerate(mapping):
    b=row['bbox_input_xyxy'];x0=max(0,int(np.floor(b[0])));x1=min(w,int(np.ceil(b[2])));y0=max(0,int(np.floor(b[1])));y1=min(h,int(np.ceil(b[3])));visible_mask[i,y0:y1,x0:x1]=255
    left=row['left_native_index'];right=row['right_native_index'];core.append(bool(redacted[left] and redacted[right]))
    for k in set([left,right]):
     if redacted[k]:
      x0,y0,x1,y1=rects[k];occluders[i,max(0,y0-3):min(h,y1+3),max(0,x0-3):min(w,x1+3)]=255
   visible_mask[occluders>0]=0
   np.savez_compressed(case/'evidence_masks.npz',masks=visible_mask,occluders=occluders,core_hidden_time=np.array(core),native_redacted=redacted,native_rectangles=rects)
   save(case/'case.json',{'sample':sample,'duration_requested_seconds':duration,'native_hidden_indices':np.flatnonzero(redacted).tolist(),'native_hidden_exposure_seconds':times[redacted].tolist(),'original_source':str(src),'core_25fps_slots':np.flatnonzero(core).tolist(),'protocol_sha256':sha(O/'PROTOCOL.json')})
   for branch in protocol['branches']:
    root=case/branch;root.mkdir()
    if branch=='rgb':
     (root/'native').mkdir();metadata=json.loads(json.dumps(native))
     for i,row in enumerate(metadata['frames']):
      original=Path(row['path']);out=root/'native'/f'{i:05d}.png'
      if sha(original)!=row['sha256']:raise ValueError('Changed original input')
      if redacted[i]:
       arr=np.array(Image.open(original).convert('RGB'));x0,y0,x1,y1=rects[i];arr[y0:y1,x0:x1]=128;Image.fromarray(arr).save(out)
       expected=np.array(Image.open(original).convert('RGB'));difference=arr!=expected;difference[y0:y1,x0:x1]=False
       if difference.any():raise ValueError('Pixels changed outside occluder')
      else:out.symlink_to(original)
      row['path']=str(out);row['sha256']=sha(out);row['synthetic_occlusion']=bool(redacted[i])
     (root/'native_cameras.npz').symlink_to(src/'native_cameras.npz');save(root/'native.json',metadata)
    else:(root/'inputs').symlink_to(src/'inputs',target_is_directory=True)
    jobs.append({'key':f'{case.name}/{branch}','root':str(root),'sample':sample,'branch':branch,'frames':61})
   pics=[]
   for i in np.flatnonzero(redacted)[::max(1,int(redacted.sum())//3)][:3]:
    im=Image.open(case/'rgb/native'/f'{i:05d}.png').convert('RGB');ImageDraw.Draw(im).text((3,3),f'native{i} synthetic',fill='white');pics.append(im)
   panel=Image.new('RGB',(w*len(pics),h));[panel.paste(im,(j*w,0)) for j,im in enumerate(pics)];panel.save(case/'occluded_input.jpg')
 save(O/'jobs.json',jobs)
 print(json.dumps({'jobs':len(jobs),'protocol_sha256':sha(O/'PROTOCOL.json')}))
if __name__=='__main__':main()
