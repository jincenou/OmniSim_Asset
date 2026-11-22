from pathlib import Path
import json,sys,hashlib,time,collections,fcntl,importlib.util,math,gc
import numpy as np
LIB=Path('/mnt/ojc/workplace1/asset_library');ROOT=LIB/'releases/rigid';OUT=Path(__file__).parent
sys.path.insert(0,'/mnt/ojc/DriveLab')
from asset_extraction.rigid_publication import validate_ply,sha256
src=Path('/mnt/ojc/workplace1/Hasim_trajectory_assets/src');spec=importlib.util.spec_from_file_location('hasim',src/'__init__.py',submodule_search_locations=[str(src)]);m=importlib.util.module_from_spec(spec);sys.modules['hasim']=m;spec.loader.exec_module(m)
from hasim.shared.assets.library_v2 import AssetLibrary,load_asset
with (LIB/'.rigid.lock').open('r') as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 text=(ROOT/'drivelab_asset.json').read_text();d=json.loads(text)
 central=json.loads((LIB/'manifests/asset_library_rigid.json').read_text())['assets']
 receipts=json.loads((LIB/'manifests/rigid_publication_receipts.json').read_text())['assets']
 library=AssetLibrary(ROOT)
(OUT/'manifest_snapshot.json').write_text(text)
index={x['global_uid']:x for x in central};assets=d['assets'];errors=[];headers=collections.Counter();required={'global_uid','dataset','class','category','asset_path','size_xyz','scale_m','scale_axis','canonical'}
def check(ok,label,uid='manifest'):
 if not ok:errors.append(dict(id=uid,check=label))
check(d['schema_version']==1 and d['manifest_type']=='drivelab_asset','schema_header')
check(d['path_policy']=='repository_relative_single_asset_path','relative_path_policy')
check(d['total']==len(assets),'total')
check(len({a['global_uid'] for a in assets})==len(assets),'unique_ids');check(len({a['asset_path'] for a in assets})==len(assets),'unique_paths')
check({a.id for a in library.assets}=={a['global_uid'] for a in assets},'actual_Hasim_catalogue_IDs')
for field,label in [('dataset','by_dataset'),('class','by_class'),('category','by_category')]:check(dict(collections.Counter(x[field] for x in assets))==d['counts'][label],'counts_'+field)
fields_required={'x','y','z','opacity',*(f'f_dc_{i}' for i in range(3)),*(f'scale_{i}' for i in range(3)),*(f'rot_{i}' for i in range(4))}
for a in assets:
 uid=a['global_uid'];check(set(a)==required,'nine_fields',uid);p=ROOT/a['asset_path'];rel=Path(a['asset_path'])
 check(not rel.is_absolute() and '..' not in rel.parts and rel.parts[:3]==('assets','rigid',a['category']) and p.resolve().is_relative_to(ROOT.resolve()),'portable_category_path',uid)
 check(p.is_file(),'payload_exists',uid)
 size=np.asarray(a['size_xyz']);axis='xyz'.index(a['scale_axis']);check(size.shape==(3,) and np.isfinite(size).all() and (size>0).all(),'size_xyz',uid);check(axis==np.argmax(size),'dominant_scale_axis',uid);check(math.isfinite(a['scale_m']) and a['scale_m']>0,'scale_m',uid)
 c=a['canonical'];rot=np.asarray(c['rotation']);tr=np.asarray(c['translation']);check(rot.shape==(3,3) and np.isfinite(rot).all() and np.allclose(rot.T@rot,np.eye(3),atol=2e-4) and abs(np.linalg.det(rot)-1)<2e-4 and tr.shape==(3,) and np.isfinite(tr).all(),'baked_transform_provenance',uid)
 if a['category'] in {'car','truck','bus','van','motorcycle','bicycle','traffic_sign','traffic_light'}:check(c.get('heading')=='+X','heading_metadata',uid)
 check(uid in receipts and receipts[uid].get('source_status')=='keep','keep_publication_receipt',uid)
 check(uid in index and index[uid]['status']=='keep','central_keep_only',uid)
 if uid in receipts:check(receipts[uid]['entry']==a,'receipt_entry_equal',uid)
 check(np.allclose(library.get(uid).dimensions,size*a['scale_m']/size[axis],atol=1e-6),'actual_loader_single_scale_dimensions',uid)
 if not p.is_file():continue
 with p.open('rb') as f:
  header=[]
  for _ in range(200):
   line=f.readline().decode('ascii').strip();header.append(line)
   if line=='end_header':break
  offset=f.tell()
 names=[l.split()[-1] for l in header if l.startswith('property ')];n=int(next(l.split()[-1] for l in header if l.startswith('element vertex ')))
 check('format binary_little_endian 1.0' in header,'binary_little_endian',uid);check(fields_required<=set(names),'gaussian_fields',uid);check(n>0,'nonempty_vertices',uid)
 if all(l.startswith('property float ') for l in header if l.startswith('property ')) and len([l for l in header if l.startswith('element ')])==1:check(p.stat().st_size==offset+n*4*len(names),'payload_byte_length',uid)
 headers[tuple(names)]+=1
# Every category plus latest outputs for every vehicle category/dataset; full numerical checks remain a sample.
samples={}
for category in sorted({a['category'] for a in assets}):
 group=[a for a in assets if a['category']==category];a=max(group,key=lambda a:(ROOT/a['asset_path']).stat().st_mtime_ns);samples[a['global_uid']]=a
for ds in ['nuscenes','waymo','web']:
 for cat in ['car','truck','van','bus']:
  group=[a for a in assets if a['dataset']==ds and a['category']==cat]
  if group:
   a=max(group,key=lambda a:(ROOT/a['asset_path']).stat().st_mtime_ns);samples[a['global_uid']]=a
sample_results=[]
for uid,a in samples.items():
 try:
  p=ROOT/a['asset_path'];digest=sha256(p);check(digest==receipts[uid]['output_sha256'],'sample_output_hash',uid)
  geom=validate_ply(p,a['size_xyz']);runtime=load_asset(library.get(uid));frame=runtime.sample(0.)
  for key in ['means','quaternions','scales','opacities','sh','local_from_appearance']:check(np.isfinite(getattr(frame,key)).all(),'runtime_finite_'+key,uid)
  check((frame.scales>0).all(),'runtime_positive_scales',uid);check(((frame.opacities>=0)&(frame.opacities<=1)).all(),'runtime_opacity_range',uid)
  sample_results.append(dict(id=uid,category=a['category'],dataset=a['dataset'],sha256=digest,gaussians=geom['gaussian_count'],sh_degree=frame.sh_degree,status='passed'));del runtime,frame;gc.collect()
 except Exception as e:errors.append(dict(id=uid,check='sample_runtime',error=repr(e)))
print('sample checks',len(sample_results),flush=True)
result=dict(checked=time.time(),passed=not errors,reference='Hasim_trajectory_assets/docs/DRIVELAB_HF_RIGID.md and rigid_release_policy.json',manifest_sha256=hashlib.sha256(text.encode()).hexdigest(),release_assets=len(assets),central_assets=len(central),central_status=dict(collections.Counter(a['status'] for a in central)),central_keep_not_in_release=sum(a['status']=='keep' and a['global_uid'] not in {r['global_uid'] for r in assets} for a in central),release_counts=d['counts'],full_scope=['schema and nine fields','unique IDs and paths','all PLY headers and byte lengths','actual Hasim catalogue and scale calculation','keep-only receipts and metadata agreement'],sample_scope='Newest per category plus newest vehicle category/dataset; full PLY numerical, hash and actual runtime load/sample checks',sample_count=len(sample_results),samples=sample_results,errors=errors,header_variants=[dict(fields=list(k),count=v) for k,v in headers.items()],limits=['Not full-library visual or physical geometry certification','Full payload hashes/numerical arrays checked only for reported sample','Human outputs not modified or revalidated in this rigid-format audit','Canonical rotation/translation already baked; apply only scale_m/size_xyz[scale_axis] once'])
(OUT/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['samples','header_variants']},ensure_ascii=False),flush=True)
