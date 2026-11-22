"""Merge the three reviewed packages into existing central and rigid-local catalogues."""
import json,hashlib,csv,io,fcntl,os,shutil,copy
from pathlib import Path
from collections import Counter
W=Path(__file__).resolve().parents[1];L=Path('/mnt/ojc/workplace1/asset_library');audit=L/'maintenance/rider_vehicle_20261030';audit.mkdir(parents=True,exist_ok=True)
NAMES={'nuscenes_b2710614':'H_rider_nuscenes_b2710614','niu_blue':'B0_rider_niu_blue','niu_black':'H_rider_niu_black'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,s):
 tmp=p.with_name('.'+p.name+'.rider-tmp');tmp.write_text(s);os.replace(tmp,p)
def write(p,x):atomic(p,json.dumps(x,ensure_ascii=False,indent=2)+'\n')
rows=[]
for name,folder in NAMES.items():
 p=L/'rigid/motorcycle'/folder;place=json.loads((p/'placement.json').read_text());quality=json.loads((p/'quality.json').read_text());assert quality['insertion_visibility']['frames']>=12
 assert quality['insertion_visibility']['edge_clipped_frames']==0
 c=json.loads((p/'transform.json').read_text());a=json.loads((p/'assembly.json').read_text());s=json.loads((W/'runs'/name/'selected/selected.json').read_text());src=s['row'];ds='nuscenes' if name.startswith('nuscenes') else 'web';uid='rider_vehicle_20261030:'+ds+':'+name;rel=str(p.relative_to(L));prov=json.loads((p/'provenance.json').read_text());dims=place['dimensions_lwh_m'];h=sha(p/'asset.ply');c.update(canonical_path=rel+'/asset.ply',sha256=h)
 row=dict(uid=uid,global_uid=uid,dataset=ds,category='motorcycle',**{'class':'motorcycle'},route=src['route'],source_path=rel+'/asset.ply',status='keep',quality_score=7,quality_scope='internal fixed-rider demo; Codex visual review, not a scan or production certification',defects=quality['limitations'],category_match=True,pose_state='fixed_seated_riding',ground_contact='two_wheels',symmetry_360_ok=True,color_saliency='bright' if name!='niu_black' else 'dark',label=dict(sub_type='seated scooter with fixed rider',usable_as=['rigid rider_vehicle'],body_brightness='dark' if name=='niu_black' else 'neutral'),scale=dict(object_desc=a['vehicle_appearance']+' and '+a['rider_appearance'],est_length_m=dims[0],est_width_m=dims[1],est_height_m=dims[2],scale_confidence='medium'),canonical=c,assembly=a,metric_scale_evidence=quality['metric_scale_evidence'],package_path=rel,storage=dict(canonical_path=rel+'/asset.ply',package_path=rel),insertion_validation=dict(status='passed_demo',mode=quality['insertion_visibility']['mode'],evidence=rel+'/quality.json'),provenance=dict(source_obj_id=name,scene_id=src['scene_id'],source_image=rel+'/source/anchor.png',source_image_sha256=sha(p/'source/anchor.png'),source=prov['source'],real_images=[dict(rgba_path=rel+f'/source/reference_{i}.png') for i in range(len(src['real_images']))],real_view_selection=prov['reference_selection'],prompts=src['prompts'],input_mode=src['input_mode'],raw_sha256=sha(p/'raw.ply'),raw_source_path=rel+'/raw.ply'),quality=dict(model='Codex visual inspection',audit=dict(keep=True,quality_score=7,scope='demo',exact_face_and_decals_verified=False,defects=quality['limitations']),assembly_structure_verified=True,wheel_support_verified=quality['support']['wheel_support_difference_m']<.05,heading_decision=c['heading'],rendering=dict(evidence=rel+'/evidence.jpg',heading_evidence=rel+'/heading.jpg',turntable=rel+'/turntable.mp4'),scale_source=quality['metric_scale_evidence']))
 # Package-local asset.json uses package-relative paths, matching existing catalogue practice.
 def local(v):
  if isinstance(v,dict):return {k:local(x) for k,x in v.items()}
  if isinstance(v,list):return [local(x) for x in v]
  return v[len(rel)+1:] if isinstance(v,str) and v.startswith(rel+'/') else v
 write(p/'asset.json',local(row));write(p/'package.json',dict(uid=uid,files={str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file() and f.name!='package.json'}));rows.append(row)
with (L/'manifests/.publish.lock').open('a') as lock,(L/'.rigid.lock').open('a') as lock2:
 fcntl.flock(lock,fcntl.LOCK_EX);fcntl.flock(lock2,fcntl.LOCK_EX)
 paths=['manifests/asset_library_rigid.json','rigid/asset_library_rigid.json','manifests/global_asset_manifest.jsonl','manifests/rigid_category_counts.json','rigid/INDEX.csv','rigid_index.html']
 for rel in paths:
  p=L/rel;b=audit/'before'/rel
  if not b.exists() and not b.with_name(b.name+'.gz').exists():b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,b)
 for rel,strip in [('manifests/asset_library_rigid.json',False),('rigid/asset_library_rigid.json',True)]:
  p=L/rel;d=json.loads(p.read_text());index={x['uid']:x for x in d['assets']}
  for row in rows:
   if row['uid'] in index:assert index[row['uid']]['canonical']['sha256']==row['canonical']['sha256']
   def translate(v):
    if isinstance(v,dict):return {k:translate(x) for k,x in v.items()}
    if isinstance(v,list):return [translate(x) for x in v]
    return v[6:] if strip and isinstance(v,str) and v.startswith('rigid/') else v
   index[row['uid']]=translate(row)
  d['assets']=list(index.values());d['total']=len(index)
  if 'counts' in d:d['counts']=dict(Counter(x['status'] for x in d['assets']))
  write(p,d)
 p=L/'manifests/global_asset_manifest.jsonl';data=[json.loads(x) for x in p.read_text().splitlines() if x.strip()];known={x.get('sha256') for x in data}
 for row in rows:
  if row['canonical']['sha256'] not in known:data.append(dict(canonical_path=row['source_path'],category='motorcycle',format='ply',sha256=row['canonical']['sha256'],size=(L/row['source_path']).stat().st_size,tier='keep',asset_kind='rider_vehicle_rigid'))
 atomic(p,''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data))
 p=L/'manifests/asset_library_rigid.json';data=json.loads(p.read_text());counts={}
 for row in data['assets']:
  c=counts.setdefault(row['category'],dict(total=0,keep=0,review=0,reject=0));c['total']+=1;c[row['status']]=c.get(row['status'],0)+1
 write(L/'manifests/rigid_category_counts.json',dict(source_manifest_sha256=sha(p),counts=counts))
 p=L/'rigid/INDEX.csv';reader=csv.DictReader(io.StringIO(p.read_text()));fields=reader.fieldnames;data=list(reader);known={x['uid'] for x in data}
 for row in rows:
  if row['uid'] not in known:
   r=row['package_path'][6:];data.append(dict(uid=row['uid'],category='motorcycle',dataset=row['dataset'],scene_id=row['provenance']['scene_id'],route=row['route'],directory=r,ply=r+'/asset.ply',metadata=r+'/asset.json'))
 out=io.StringIO();writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader();writer.writerows(data);atomic(p,out.getvalue())
 p=L/'rigid_index.html';s=p.read_text();banner='<aside id="rider-vehicle-demos" style="padding:16px;background:#eaf4ff;color:#16324f"><a href="previews/rider_vehicle_demos/index.html">新增：3 个固定骑姿人车组合 demo（2 个电动踏板车 + 1 个非电动对照）</a></aside>'
 if 'id="rider-vehicle-demos"' not in s:
  i=s.lower().find('<body');j=s.find('>',i);s=s[:j+1]+banner+s[j+1:];atomic(p,s)
 write(audit/'publication.json',dict(added_uids=[x['uid'] for x in rows],packages=[x['package_path'] for x in rows],central_total=json.loads((L/'manifests/asset_library_rigid.json').read_text())['total'],existing_rows_preserved=True))
print('PUBLISHED',len(rows),'into',L/'rigid/motorcycle')
