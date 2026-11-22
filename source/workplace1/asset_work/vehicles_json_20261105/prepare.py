"""Exact NuScenes instance extraction; annotations locate inputs, never replace identity."""
import collections, hashlib, itertools, json, time
from pathlib import Path
import ijson
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=Path('/mnt/public_data/nuScenes')
INDEX=ROOT.parent/'scene_production_20261031/inputs/nuscenes/debug/dataset_scenes'
def read(p): return json.loads(Path(p).read_text())
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def rot(q):
 w,x,y,z=np.array(q)/np.linalg.norm(q)
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def main():
 source=Path('/mnt/ojc/workplace1/vehicles.json'); rows=read(source)
 assert len({(r['scene_token'],r['instance_token']) for r in rows})==len(rows)
 write(ROOT/'request.json',dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),assets=rows))
 cats={r['token']:r['name'] for r in read(DATA/'v1.0-trainval/category.json')}
 instances={r['token']:cats[r['category_token']] for r in read(DATA/'v1.0-trainval/instance.json')}
 scenes={r['token']:r['name'] for r in read(DATA/'v1.0-trainval/scene.json')}
 samples={r['token']:r['scene_token'] for r in read(DATA/'v1.0-trainval/sample.json')}
 wanted={r['instance_token'] for r in rows}; annotations=collections.defaultdict(list)
 with (DATA/'v1.0-trainval/sample_annotation.json').open('rb') as f:
  for a in ijson.items(f,'item',use_float=True):
   if a['instance_token'] in wanted: annotations[a['instance_token']].append(a)
 indexes={}; output=[]; signs=np.array(list(itertools.product([-1,1],repeat=3)))
 for n,r in enumerate(rows):
  uid='vehicles_json_20261105__'+r['scene_name']+'__'+r['instance_token']
  rec=dict(r,obj_id=uid,pipeline='human' if r['category'].startswith('human.') else 'rigid_H')
  try:
   assert scenes[r['scene_token']]==r['scene_name'], 'scene token/name mismatch'
   assert instances[r['instance_token']]==r['category'], 'instance category mismatch'
   if rec['pipeline']=='human':raise ValueError('Human requires separate human pipeline, not rigid fallback')
   if r['category'] not in {'vehicle.car','vehicle.truck','vehicle.bus.rigid'}:raise ValueError('Unsupported category; no generic fallback')
   rec['asset_category']=r['category'].split('.')[1]
   if r['scene_name'] not in indexes:
    ix=read(INDEX/(r['scene_name']+'.json'));assert ix['scene_token']==r['scene_token']
    bysample=collections.defaultdict(list)
    for f in ix['frames']:
     f={k:v for k,v in f.items() if k!='motion_paths'}
     f['_inverse']=np.linalg.inv(f['camera_to_world']);bysample[f['sample_token']].append(f)
    indexes[r['scene_name']]=bysample
   candidates=[]
   for a in annotations[r['instance_token']]:
    if samples[a['sample_token']]!=r['scene_token']:continue
    size=[a['size'][1],a['size'][0],a['size'][2]]
    world=(signs*np.array(size)/2)@rot(a['rotation']).T+np.array(a['translation'])
    for f in indexes[r['scene_name']].get(a['sample_token'],[]):
     tf=f['_inverse'];xyz=world@tf[:3,:3].T+tf[:3,3]
     if xyz[:,2].min()<=.1:continue
     uv=xyz@np.array(f['camera_intrinsic']).T;uv=uv[:,:2]/uv[:,2:]
     low=uv.min(0);high=uv.max(0)
     # All native camera images are 1600x900. Verify actual dimensions during segmentation.
     if (low<3).any() or (high>[1597,897]).any():continue
     width,height=high-low
     # Native visibility is not a per-camera segmentation verdict. Keep it as
     # ranking evidence; downstream image review determines visible completeness.
     if width<40 or height<30:continue
     if not Path(f['path']).is_file():continue
     frame={k:v for k,v in f.items() if k!='_inverse'}
     score=float(width*height)*(1 if a['visibility_token']=='4' else .65)
     candidates.append(dict(frame=frame,annotation_token=a['token'],bbox_xyxy=[*low.tolist(),*high.tolist()],size_lwh_m=size,visibility_token=a['visibility_token'],score=score))
   candidates.sort(key=lambda v:-v['score']); chosen=[]
   for c in candidates:
    if any(c['frame']['camera_name']==d['frame']['camera_name'] and abs(c['frame']['timestamp_us']-d['frame']['timestamp_us'])<500000 for d in chosen):continue
    chosen.append(c)
    if len(chosen)==6:break
   rec.update(status='source_candidates_ready' if chosen else 'no_complete_visible_source',candidates=chosen,eligible_observations=len(candidates))
  except Exception as e:rec.update(status='source_error',error=repr(e),candidates=[])
  write(ROOT/'sources'/uid/'candidates.json',rec);output.append(rec)
  if n%25==0:print(n+1,len(rows),rec['status'],flush=True)
 write(ROOT/'inventory.json',dict(status='prepared',total=len(rows),counts=dict(collections.Counter(r['status'] for r in output)),categories=dict(collections.Counter(r['category'] for r in output)),assets=output,updated=time.time()))
 print('Prepared',len(output),collections.Counter(r['status'] for r in output),flush=True)
if __name__=='__main__':main()
