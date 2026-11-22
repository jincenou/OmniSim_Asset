"""Read-only evaluation on hidden native pixels, never on training detections."""
from pathlib import Path
import argparse,json,sys,csv,hashlib
import numpy as np
import torch,av
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation,Slerp
from safetensors.torch import load_file
G=Path(__file__).resolve().parent;W=G.parent;O=G/'occlusion_v1';REPO=W.parent/'repo'
sys.path.insert(0,str(REPO))
from fdanyone.skeleton.pipeline import _gvhmr_geometry_context
from review_motion import EDGES

def interpolate_rot(t,a,q):
 shape=a.shape;flat=a.reshape(len(t),-1,3);answer=np.stack([Slerp(t,Rotation.from_rotvec(flat[:,j]))(q).as_rotvec() for j in range(flat.shape[1])],axis=1)
 return torch.tensor(answer.reshape(len(q),*shape[1:]),dtype=torch.float32)
def interpolate_params(params,t,q):
 out={}
 for k,a in params.items():
  x=a.numpy()
  if k in ['body_pose','global_orient']:out[k]=interpolate_rot(t,x,q)
  else:out[k]=torch.tensor(np.stack([np.interp(q,t,x[:,j]) for j in range(x.shape[1])],1),dtype=torch.float32)
 return out

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--include-rohm',action='store_true');args=parser.parse_args()
 torch.set_num_threads(2);rows=[];media=[]
 jobs=json.loads((O/'jobs.json').read_text());cases=sorted(set(str(Path(j['root']).parent) for j in jobs))
 with _gvhmr_geometry_context(REPO/'third_party/GVHMR'):
  from hmr4d.utils.smplx_utils import make_smplx
  model=make_smplx('supermotion_coco17').eval()
  for casepath in cases:
   case=Path(casepath);meta=json.loads((case/'case.json').read_text());sample=meta['sample'];src=Path(meta['original_source']);native=json.loads((src/'native.json').read_text());camera=np.load(src/'native_cameras.npz')
   reference=np.load(O/'evaluation_reference'/sample/'native_detections.npz');kp=reference['keypoints'];boxes=reference['boxes'];native_t=reference['native_timestamps']-reference['native_timestamps'][0];masks=np.load(case/'evidence_masks.npz');native_hidden=masks['native_redacted'];rect=masks['native_rectangles']
   xy=kp[...,:2];inside=(xy[...,0]>=rect[:,None,0])&(xy[...,0]<rect[:,None,2])&(xy[...,1]>=rect[:,None,1])&(xy[...,1]<rect[:,None,3])
   bbox_inside=(xy[...,0]>=boxes[:,None,0])&(xy[...,0]<=boxes[:,None,2])&(xy[...,1]>=boxes[:,None,1])&(xy[...,1]<=boxes[:,None,3]);confident=(kp[...,2]>=.5)&bbox_inside
   hidden=inside&native_hidden[:,None]&confident;leg_mask=np.isin(np.arange(17),[11,12,13,14,15,16])[None,:]
   shown=confident&~(inside&native_hidden[:,None]);height=boxes[:,3]-boxes[:,1]
   projections={};metrics={};params_all={};t=None
   for branch in ['rgb','kp_only']:
    root=case/branch;old=load_file(str(root/'motion/motion.safetensors'));new=load_file(str(root/'optimized/motion_candidates.safetensors'));t=new['times_seconds'].numpy()
    in_range=(native_t>=t[0])&(native_t<=t[-1]);q=native_t[in_range]
    raw={k:old['smpl_params_incam.'+k] for k in ['body_pose','betas','global_orient','transl']};raw['betas']=new['betas']
    variants={'O0':raw,'O1':dict(raw,body_pose=new['proposal_initial.body_pose']),
       'O2':dict(raw,body_pose=new['body_pose'],global_orient=new['incam.global_orient'],transl=new['incam.transl'])}
    if args.include_rohm and branch=='rgb':
     o3=load_file(str(root/'optimized_rohm/motion_candidates.safetensors'))
     variants['O3']=dict(raw,body_pose=o3['body_pose'],global_orient=o3['incam.global_orient'],transl=o3['incam.transl'])
    for variant,params in variants.items():
     name=f'{branch}_{variant}';params_all[name]=params
     with torch.inference_mode():points=model(**interpolate_params(params,t,q)).numpy();points25=model(**params).numpy()
     if (points[...,2]<=0).any():raise ValueError('Behind-camera evaluated body')
     homogeneous=np.einsum('tij,tnj->tni',camera['K_input'][in_range],points);uv=homogeneous[...,:2]/homogeneous[...,2:];projections[name]=uv
     error=np.linalg.norm(uv-xy[in_range],axis=-1)/height[in_range,None]
     metrics[name]={}
     for region,mask in [('hidden',hidden),('hidden_legs',hidden&leg_mask),('visible',shown)]:
      v=error[mask[in_range]];metrics[name][region]={'n':int(len(v)),'nme_mean':float(v.mean()) if len(v) else None,'nme_p95':float(np.percentile(v,95)) if len(v) else None}
     R=Rotation.from_rotvec(params['global_orient'].numpy()).as_matrix();local=np.einsum('tji,tnj->tni',R,points25-params['transl'].numpy()[:,None,:]);local-=local[:,11:13].mean(1,keepdims=True)
     scale=json.loads((root/'optimized/report.json').read_text())['body_scale_m'];dt=np.diff(t);velocity=np.diff(local,axis=0)/dt[:,None,None];accel=np.diff(velocity,axis=0)/((dt[1:]+dt[:-1])/2)[:,None,None]
     metrics[name]['estimated_local_range']=float(np.sqrt(np.mean(np.ptp(local,axis=0)**2))/scale);metrics[name]['estimated_local_accel']=float(np.sqrt(np.mean(accel**2))/scale)
     rows.append({'case':case.name,'sample':sample,'branch':branch,'variant':variant,'duration_requested_s':meta['duration_requested_seconds'],
       **{f'{r}_{k}':v for r in ['hidden','hidden_legs','visible'] for k,v in metrics[name][r].items()},'estimated_local_range':metrics[name]['estimated_local_range'],'estimated_local_accel':metrics[name]['estimated_local_accel']})
   # One consolidated native-time media per duration; hidden originals are for evaluator display only.
   w,h=native['size'];panels=[];columns=['native heldout','synthetic occluder','rgb_O0','rgb_O1','rgb_O2','rgb_O3' if args.include_rohm else 'kp_only_O2'];indices=np.flatnonzero(in_range)
   for qi,i in enumerate(indices):
    original=Image.open(native['frames'][i]['path']).convert('RGB');panel=Image.new('RGB',(w*6,h+24),'#222222');label=ImageDraw.Draw(panel)
    for ci,name in enumerate(columns):
     im=Image.open(case/'rgb/native'/f'{i:05d}.png').convert('RGB') if ci==1 else original.copy();d=ImageDraw.Draw(im)
     if ci>=2:
      uv=projections[name][qi]
      for a,b in EDGES:d.line([tuple(uv[a]),tuple(uv[b])],fill='lime',width=2)
      for j,(x,y,score) in enumerate(kp[i]):
       if confident[i,j]:d.ellipse((x-2,y-2,x+2,y+2),fill='red' if hidden[i,j] else 'yellow')
     panel.paste(im,(ci*w,24));label.text((ci*w+3,4),f'{name} n{i}',fill='white')
    panels.append(panel)
   output=case/'evaluation';output.mkdir(exist_ok=args.include_rohm)
   with av.open(str(output/'comparison.mp4'),'w') as file:
    stream=file.add_stream('libx264',rate=10);stream.width=w*6;stream.height=h+24;stream.pix_fmt='yuv420p';stream.options={'crf':'19','threads':'2'}
    for panel in panels:
     for packet in stream.encode(av.VideoFrame.from_image(panel)):file.mux(packet)
    for packet in stream.encode():file.mux(packet)
   candidates=np.flatnonzero(native_hidden[in_range]);slots=sorted(set([int(candidates[0]),int(candidates[len(candidates)//2]),int(candidates[-1])]))
   contact=Image.new('RGB',(w*6,(h+24)*len(slots)))
   for r,slot in enumerate(slots):contact.paste(panels[slot],(0,r*(h+24)))
   contact.save(output/'comparison.jpg',quality=94)
   with av.open(str(output/'comparison.mp4')) as c:assert sum(1 for _ in c.decode(video=0))==len(indices)
   report={'case':case.name,'metrics':metrics,'native_times_evaluated':native_t[in_range].tolist(),'native_indices_excluded_outside_estimate_support':np.flatnonzero(~in_range).tolist(),
     'fixed_hidden_count':int(hidden[in_range].sum()),'fixed_hidden_leg_count':int((hidden&leg_mask)[in_range].sum()),'includes_fixed_RoHM_proposal':args.include_rohm,'metrics_reference':'ViTPose on actual unoccluded native pixels; not human 2D annotation or 3D ground truth',
     'prediction_at_native_time':'SO3 interpolation of camera-space root and local rotations, linear camera translation, with native crop K; no extrapolation',
     'source_frame_observation_scope':'native frames6..18 visually reviewed for lower-body visibility; 2D semantic labels remain detector-derived',
     'video_frames_decoded':len(indices),'playback_rate':10,'quality_passed':False,'colors':'green fitted; red hidden reference detections; yellow visible reference detections',
     'feature_provenance':{b:json.loads((case/b/'motion_pair/feature_provenance.json').read_text()) for b in ['rgb','kp_only']}}
   (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');media.append(case.name)
 with (O/'metrics.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 (O/'metrics.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
 (O/'evaluation_status.json').write_text(json.dumps({'state':'complete_not_quality_approved','includes_rohm':args.include_rohm,'cases':media,'counterfactual_limitations':'Synthetic known occlusion and detector evaluation; not recovery of real hidden truth','updated_utc_seconds':__import__('time').time()},indent=2)+'\n')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
