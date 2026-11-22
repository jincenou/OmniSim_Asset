"""Independent immutable manifest + training-only time-stamped visual hull initialization."""
import argparse,hashlib,json,math,os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'waymo/round2'

def output_path(path):
 p=Path(path).resolve()
 if not p.is_relative_to(ROOT):raise ValueError(f'Output escapes experiment root: {p}')
 p.parent.mkdir(parents=True,exist_ok=True)
 return p

def save_json(path,data):
 p=output_path(path);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,p)

def sha(path):
 p=Path(path);s=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 e=p.stat()
 if (s.st_size,s.st_mtime_ns)!=(e.st_size,e.st_mtime_ns):raise RuntimeError(f'Source changed: {p}')
 return h.hexdigest()

def project(points,c):
 w=np.linalg.inv(c['c2w']);v=points@w[:3,:3].T+w[:3,3];q=v@np.array(c['K']).T
 uv=q[:,:2]/q[:,2:];ij=np.rint(uv).astype(int)
 valid=(v[:,2]>0)&(ij[:,0]>=0)&(ij[:,0]<c['width'])&(ij[:,1]>=0)&(ij[:,1]<c['height'])
 return ij,valid

def prepare(label,object_id):
 dest=output_path(ROOT/'objects'/object_id/'clips/C001/manifest.json')
 if dest.exists():raise FileExistsError(dest)
 job=json.loads((OLD/'demo_jobs'/f'{label}.json').read_text());source=Path(job['source']);data=Path(job['data'])
 cam_src=json.loads((source/'cameras.json').read_text());mapping=json.loads(Path(job['mapping']).read_text())
 assert cam_src['world_frame']['name']=='root_local_human_world'
 assert len(mapping['frames'])==61
 catalog=json.loads((ROOT.parent/'waymo/objects/catalog.json').read_text())
 entries=catalog if isinstance(catalog,list) else catalog['objects']
 identity=next(x for x in entries if x['object_id']==object_id)
 cameras=[];hashes={}
 files=[OLD/'demo_jobs'/f'{label}.json',source/'cameras.json',source/'metadata.json',source/'root_motion.npz',source/'root_local.json',Path(job['mapping']),Path(job['native_mapping']),data/'sparse/0/cameras.txt',data/'sparse/0/images.txt']
 for p in files:hashes[str(p.resolve())]=sha(p)
 intr={int(t[0]):list(map(float,t[4:])) for line in (data/'sparse/0/cameras.txt').read_text().splitlines() if (t:=line.split()) and not line.startswith('#')}
 extr={}
 for line in (data/'sparse/0/images.txt').read_text().splitlines():
  t=line.split()
  if not t or line.startswith('#'):continue
  q=np.array(t[1:5],float);w=np.eye(4);w[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();w[:3,3]=np.array(t[5:8],float);extr[int(t[0])]=np.linalg.inv(w)
 for c in cam_src['cameras']:
  cid=c['camera_id'];p=data/'images'/f'cam{cid:02d}'/'000000.png'
  with Image.open(p) as im:width,height=im.size;assert im.mode=='RGBA'
  K=np.array(c['K']);K[0]*=width/c['image_width'];K[1]*=height/c['image_height']
  assert np.allclose([K[0,0],K[1,1],K[0,2],K[1,2]],intr[cid+1],atol=1e-6)
  assert np.allclose(c['camera_to_world'],extr[cid+1],atol=1e-6)
  cameras.append({'id':cid,'yaw':c['yaw'],'pitch':c['pitch'],'K':K.tolist(),'c2w':c['camera_to_world'],'width':width,'height':height})
  for f in range(61):
   p=data/'images'/f'cam{cid:02d}'/f'{f:06d}.png';hashes[str(p.resolve())]=sha(p)
 assert sorted(c['id'] for c in cameras)==list(range(24))
 val=[0,8,16];available=[c for c in cameras if c['id'] not in val]
 def select(n):
  picked=[]
  for angle in np.arange(n)*360/n+15:
   c=min([c for c in available if c['id'] not in picked],key=lambda c:abs((c['yaw']-angle+180)%360-180));picked.append(c['id'])
  return picked
 seconds=[f['video_seconds'] for f in mapping['frames']];assert np.allclose(seconds,np.arange(61)/25)
 protocol={name:{'train_camera_ids':select(n),'train_frame_ids':list(range(0,61,stride)),'init_frame_ids':frames,'views_per_step':1,'times_per_step':1,'microbatch':1,'gradient_accumulation':4} for name,n,stride,frames in [('S0',8,5,[0,15,30,45,60]),('S1',12,2,[0,10,20,30,40,50,60])]}
 out={'schema':1,'object_id':object_id,'label':label,'identity':identity,'source_job':job,'data_root':str(data),'cameras':cameras,'available_camera_ids':list(range(24)),'validation_camera_ids':val,'available_frame_ids':list(range(61)),'evaluation_frame_ids':list(range(61)),'render_timestamps_sec':seconds,'frame_mapping':mapping['frames'],'real_seconds_to_model_time':{'scale':1/2.4,'offset':0},'model_time_to_real_seconds':{'scale':2.4,'offset':0},'source_real_frame_count':25,'direct_source_slots':sum(not f['interpolated'] for f in mapping['frames']),'source_interpolated_slots':sum(f['interpolated'] for f in mapping['frames']),'sample_span_seconds':2.4,'model_time_duration':[0,1],'coordinates':{'training_frame':'root_local_human_world','source_to_training':np.eye(4).tolist(),'training_to_source':np.eye(4).tolist(),'root_transform_source':str(source/'root_motion.npz'),'metric_scale_verified':False},'protocols':protocol,'source_hashes':hashes,'comparison':'Old sigma8/15k uses21x61; S1 uses12x31, reference only, no pure backend ranking.'}
 # Recheck all sources after import. No symlinks or writes to old data.
 for p,h in hashes.items():assert sha(p)==h,p
 save_json(dest,out)
 for sub in ['01_input','02_preprocess','03_motion','04_generation','05_reconstruction','06_assets','07_demo','08_evaluation','09_records']:(dest.parent/sub).mkdir(exist_ok=True)
 save_json(dest.parent/'09_records/source_checks.json',{'verified_before_after':True,'file_count':len(hashes),'manifest_sha256':sha(dest)})
 print(dest,flush=True)
 return dest

def initialize(manifest,protocol):
 m=json.loads(Path(manifest).read_text());p=m['protocols'][protocol];cs=[c for c in m['cameras'] if c['id'] in p['train_camera_ids']]
 assert not set(p['train_camera_ids'])&set(m['validation_camera_ids'])
 assert set(p['init_frame_ids'])<=set(p['train_frame_ids'])
 out=output_path(Path(manifest).parent/'05_reconstruction'/f'init_{protocol}.npz')
 if out.exists():raise FileExistsError(out)
 shape=(64,96,64);grid=np.stack(np.meshgrid(np.linspace(-1.2,1.2,64),np.linspace(-.1,2.1,96),np.linspace(-1.2,1.2,64),indexing='ij'),-1).reshape(-1,3)
 projections=[project(grid,c) for c in cs];pts=[];colors=[];times=[];counts=[]
 for frame in p['init_frame_ids']:
  ims=[]
  for c in cs:
   f=Path(m['data_root'])/'images'/f"cam{c['id']:02d}"/f'{frame:06d}.png';assert sha(f)==m['source_hashes'][str(f.resolve())];ims.append(np.array(Image.open(f)))
  votes=np.zeros(len(grid),np.int16)
  for im,(uv,valid) in zip(ims,projections):
   hit=np.zeros(len(grid),bool);hit[valid]=im[uv[valid,1],uv[valid,0],3]>=128;votes+=hit
  occ=(votes>=math.ceil(.95*len(cs))).reshape(shape)
  if any(face.any() for face in [occ[0],occ[-1],occ[:,0],occ[:,-1],occ[:,:,0],occ[:,:,-1]]):raise RuntimeError('Hull touches boundary')
  xyz=grid[np.flatnonzero((occ&~binary_erosion(occ)).ravel())];assert len(xyz)>50
  rgb=np.zeros((len(xyz),3));weight=np.zeros(len(xyz))
  for im,c in zip(ims,cs):
   uv,v=project(xyz,c);v[v]&=im[uv[v,1],uv[v,0],3]>=128;rgb[v]+=im[uv[v,1],uv[v,0],:3]/255;weight[v]+=1
  assert weight.min()>0
  pts.append(xyz);colors.append(rgb/weight[:,None]);times.append(np.full((len(xyz),1),model_time(m,observation_seconds(m,frame))));counts.append({'frame':frame,'points':len(xyz)})
 np.savez_compressed(out,points=np.concatenate(pts).astype('float32'),colors=np.concatenate(colors).astype('float32'),time=np.concatenate(times).astype('float32'))
 save_json(out.with_suffix('.json'),{'method':'training-only time-stamped visual hull surface, local adaptation','protocol':p,'point_counts':counts,'total_points':sum(x['points'] for x in counts),'grid_shape':shape,'bounds':[[-1.2,1.2],[-.1,2.1],[-1.2,1.2]],'vote_threshold':.95,'geometry_npz_contains_scales':False,'runtime_initial_spatial_sigma':.022,'runtime_initial_time_sigma_model':.15,'reason':'Explicit local runtime overrides spatial-only KNN for repeated spatial points across times','manifest_sha256':sha(manifest),'init_sha256':sha(out)})
 print(out,counts,flush=True)

# Shared explicit clock helpers. Existing manifests retain their exact[0,1] mapping.
def observation_seconds(manifest,frame):
 records={int(r['slot']):r for r in manifest['frame_mapping']}
 return float(records[int(frame)]['video_seconds'])

def model_time(manifest,seconds):
 transform=manifest['real_seconds_to_model_time']
 return float(seconds)*transform['scale']+transform['offset']

if __name__=='__main__':
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--object',choices=['R07','M02']);a.add_argument('--manifest',type=Path);a.add_argument('--initialize');a.add_argument('--prepare',action='store_true');args=a.parse_args()
 if args.manifest:
  m=args.manifest
  if args.prepare:a.error('--prepare uses --object, not --manifest')
 else:
  if not args.object:a.error('--object or --manifest required')
  label,oid={'R07':('r07_short61_24','OBJ_0018'),'M02':('m02_short61_24','OBJ_0019')}[args.object];m=ROOT/'objects'/oid/'clips/C001/manifest.json'
 if args.prepare:prepare(label,oid)
 if args.initialize:initialize(m,args.initialize)

