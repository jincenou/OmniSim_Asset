from pathlib import Path
import json,hashlib,datetime,sys,shutil
import numpy as np
from PIL import Image,ImageDraw
D=Path(__file__).resolve().parents[1];ROOT=D.parent;OLD=ROOT/'early_visual_query_20261019';BASE=ROOT.parent/'random8/r20261015_113332'
sys.path.insert(0,str(ROOT/'scripts'))
from roi_candidates import candidates,project,COLORS

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
assert not (D/'PLAN_DIAGNOSTIC.json').exists()
plan=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='fixed checkpoint diagnostic; zero optimizer calls; no training',headroom_GiB=5,modes={'M00':[0,0],'M10':[1,0],'M01':[0,1],'M11':[1,1]},tolerance=1e-5,heldout=[0,8,16],support_views=[1,13,7,19],selections={},descriptive_parameters={'saturation_fraction_of_component_bound':.95,'covariance_epsilon_relative_L2':1e-8,'ill_condition_min_eigen_relative_L2':1e-12,'ill_condition_max':1e10,'contribution_sum_relative_tolerance':1e-4,'covariance_probe_rgb_tolerance':1e-5,'support_depth_ambiguity_relative_L':.01,'top_display_gaussians':8,'token_label':'reference skeleton proxy; no material truth; unknown preserved'},limitations=['H is historically observed development reference','posthoc mode switch is not a training causal ablation','M00 is not R0','no new quality acceptance gates'],source_files={},roi_hashes={})
for name in ['REPORT.md','FINDINGS.md','METRICS.md','ASSETS.json','PLAN.json','gate_results.json','expanded_summary.csv','visual_review.json','NEXT_STEP.md']:
 p=OLD/name
 if p.exists():plan['source_files'][str(p)]=sha(p)
for p in (OLD/'scripts').glob('*.py'):plan['source_files'][str(p)]=sha(p)
quality=ROOT/'optimization_quality_v3_20261018/FINDINGS.md';plan['source_files'][str(quality)]=sha(quality)
inv=[]
for label,n in [('N06',61),('N11',121)]:
 times=np.rint(np.linspace(0,n-1,13)).astype(int).tolist();view,lo,hi=(16,22,46) if label=='N06' else (0,81,105)
 obs=sorted(set([(v,f) for v in [0,8,16] for f in times]+[(view,f) for f in range(lo,hi+1)]))
 deps=sorted(set(obs+[(v,f-1) for v,f in obs if f>0]));fs=sorted(set(f for v,f in obs))
 plan['selections'][label]=dict(frames=n,global_times=times,observations=obs,render_with_previous_dependencies=deps,support_times=fs,window=dict(view=view,start=lo,end=hi),last_frame_views=[0,8,16],foot_note='Old last-frame fullbody crops; no previously stored precise foot crop; no new favorable crop')
 out=D/'data'/label;out.mkdir(exist_ok=True)
 sample=BASE/'samples'/label;pts=np.load(sample/'generation/preflight/conditioning/keypoints_3d.npy');cams=json.loads((sample/'generation/outputs/turbo/cameras.json').read_text())['cameras']
 proxy_rows=[];contact=Image.new('RGB',(352*4,640*3));draw=ImageDraw.Draw(contact)
 for vi,v in enumerate([1,13,7,19]):
  record=next(c for c in cams if c['camera_id']==v);K=np.array(record['K'],float);K[:2]*=.5
  all_labels=[]
  existing=ROOT/'optimization/data'/label/f'view{v:02d}.npy'
  oldlab=np.load(existing) if existing.exists() else None
  for f in fs:
   rgba=np.array(Image.open(sample/'data/images'/f'cam{v:02d}'/f'{f:06d}.png'));uv,depth=project(pts[f],record['camera_to_world'],K);roi,support=candidates(uv,rgba[...,3]>127)
   if oldlab is not None:roi=oldlab[f].copy()
   all_labels.append(roi);proxy_rows.append(dict(view=v,frame=f,origin=str(existing) if oldlab is not None else 'unchanged roi_candidates reference-only rules',counts=[int((roi==i).sum()) for i in range(7)],centerline_support=support,visibility='ambiguous and unassigned remain unknown; depth diagnostic further gates read classification'))
   reps=[fs[0],fs[len(fs)//2],fs[-1]]
   if f in reps:
    rgb=rgba[...,:3]*(rgba[...,3:]/255)+127.5*(1-rgba[...,3:]/255)
    for k,c in enumerate(COLORS):rgb[roi==k+1]=.55*rgb[roi==k+1]+.45*c
    im=Image.fromarray(rgb.astype('uint8'));ImageDraw.Draw(im).text((5,5),f'{label} v{v} f{f} PROXY',fill='white');contact.paste(im,(vi*352,reps.index(f)*640))
  p=out/f'support{v:02d}.npz';np.savez_compressed(p,frames=fs,labels=np.stack(all_labels));plan['roi_hashes'][str(p)]=sha(p)
 contact.save(out/'support_proxy.png');save(out/'support_proxy.json',proxy_rows)
 for v in [0,8,16]:
  for p in [OLD/'data/expanded_roi'/label/f'view{v:02d}.npz',ROOT/'audit/roi'/label/f'view{v:02d}.npz',ROOT/'audit/roi'/label/f'flow{v:02d}.npz']:plan['roi_hashes'][str(p)]=sha(p)
 for arm in ['R1','R2','R3']:
  asset=OLD/'assets'/f'{label}_{arm}';meta=json.loads((asset/'asset.json').read_text());inv.append(dict(key=f'{label}_{arm}',asset=str(asset),step=meta['actual_step'],points=meta['points'],before={str(p):sha(p) for p in asset.iterdir() if p.is_file()}))
# Entire old unknown inventory is retained even if diagnostic subset is smaller.
unknown=[]
for v in [0,8,16]:
 with np.load(OLD/'data/expanded_roi/N11'/f'view{v:02d}.npz') as z:
  unknown.extend(dict(view=v,frame=int(f),side=0) for f in np.flatnonzero(~z['known'][:,0]))
assert len(unknown)==20
save(D/'data/N11/old_left_unknown.json',unknown)
plan['old_N11_left_unknown_rows']=20
save(D/'checkpoint_inventory.json',inv);save(D/'PLAN_DIAGNOSTIC.json',plan);save(D/'STATUS.json',dict(state='plan_frozen_ready_for_readonly_smoke',training_started=False,optimizer_calls=0));print('Frozen',len(inv),'assets; free GiB',shutil.disk_usage(D).free/2**30)
