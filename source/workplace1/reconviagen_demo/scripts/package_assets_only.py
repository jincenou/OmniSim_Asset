"""User-requested 20 identities x H/R/B0; only canonical Gaussian assets and metadata."""
from pathlib import Path
import json,csv,io,hashlib,zipfile,shutil
R=Path(__file__).resolve().parents[1];D=R/'deliveries';name='assets_20objects_H_R_B0_20261023';archive=D/(name+'.zip');tmp=archive.with_suffix('.zip.partial')
variants=['H','R','B0'];groups=[('vehicles',R,R/'inputs/random10_cases.json'),('other_objects',R/'nonvehicle10',R/'nonvehicle10/inputs/cases.json')]
def sha_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
# Preflight every requested asset before retiring the older deliveries.
items=[]
for group,root,casefile in groups:
 cases=json.loads(casefile.read_text());assert len(cases)==10
 for v in variants:
  for c in cases:
   cid=c['case_id'];src=root/'results'/cid/f'{v}_seed42'/'asset';assert (src/'canonical.ply').is_file();assert (src/'metadata.json').is_file()
   items.append((v,group,c,src))
assert len(items)==60
# Exact allowlist from the user's selected old deliveries. Original results/checkpoints are untouched.
removed=[]
for old in ['insertion_handoff_20261023','insertion_backgrounds_20261023','insertion_frames_20261023']:
 for suffix in ['.zip','.zip.sha256']:
  p=D/(old+suffix)
  if p.exists():
   assert p.parent.resolve()==D.resolve() and not p.is_symlink();size=p.stat().st_size;p.unlink();removed.append(dict(path=str(p),bytes=size))
p=D/'insertion_handoff_20261023'
if p.exists():
 assert p.parent.resolve()==D.resolve() and not p.is_symlink();size=sum(x.stat().st_size for x in p.rglob('*') if x.is_file());shutil.rmtree(p);removed.append(dict(path=str(p),bytes=size))
(R/'docs/assets_only_cleanup.json').write_text(json.dumps(dict(reason='User explicitly requested deletion of the three obsolete insertion delivery packages and replacement with H/R/B0 assets only',removed=removed,total_bytes=sum(x['bytes'] for x in removed),original_experiment_outputs_preserved=True,original_background_checkpoints_preserved=True),indent=2))
print('REMOVED_BYTES',sum(x['bytes'] for x in removed),flush=True)
rows=[];hashes=[]
with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
 for v,group,c,src in items:
  cid=c['case_id'];base=f'{v}/{group}/{cid}';ply=src/'canonical.ply';ph=sha_file(ply);meta=json.loads((src/'metadata.json').read_text())
  if 'canonical' in meta:meta['canonical']['canonical_path']=base+'/canonical.ply'
  encoded=(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();mh=hashlib.sha256(encoded).hexdigest()
  z.write(ply,name+'/'+base+'/canonical.ply');z.writestr(name+'/'+base+'/metadata.json',encoded)
  hashes.extend([(ph,base+'/canonical.ply'),(mh,base+'/metadata.json')])
  # Header verification without altering or decoding the existing Gaussian payload.
  header=[]
  with ply.open('rb') as f:
   for _ in range(1000):
    line=f.readline().decode('ascii').strip();header.append(line)
    if line=='end_header':break
   else:raise ValueError('Invalid PLY header '+str(ply))
  n=int(next(s.split()[-1] for s in header if s.startswith('element vertex ')));assert n>0
  for field in ['f_dc_0','opacity','scale_0','rot_0']:assert any(s.endswith(' '+field) for s in header),field
  row=dict(route=v,group=group,case_id=cid,category=c.get('category','car'),ply=base+'/canonical.ply',metadata=base+'/metadata.json',gaussian_count=n,ply_bytes=ply.stat().st_size,ply_sha256=ph,metadata_sha256=mh,metric_scale_known=group=='vehicles',reference_length_m=meta.get('scale',{}).get('est_length_m'),original_ply=str(ply))
  rows.append(row);print('ADDED',v,group,cid,flush=True)
 readme='''资产输出包：10个车辆 + 10个其他物体 × H / R / B0，共60份原生3D Gaussian PLY。

目录：
H/vehicles/                10辆车
H/other_objects/           10个其他物体
R/vehicles/                10辆车
R/other_objects/           10个其他物体
B0/vehicles/               10辆车
B0/other_objects/          10个其他物体
每个对象文件夹只有 canonical.ply 和 metadata.json；三条路线使用相同的20个对象，均seed42。

INDEX.csv / INDEX.json为完整清单，路径均相对本包根目录。SHA256SUMS.txt用于校验。
H：真实图 + 同实例FLUX → ReconViaGen；R：真实图 → ReconViaGen；B0：FLUX → 原TRELLIS。
车辆B0使用两张FLUX；本批其他物体B0使用一张FLUX。不同对象的可靠真实图数可能不同。

PLY是原始实验导出的标准化Gaussian，字节未修改；不是普通点云或mesh。
编码为SH0、log-scale、opacity logit、wxyz四元数。canonical PLY已经施加标准化；metadata内rotation/translation是过程记录，不要再次施加。
车辆保留尺寸估计，但canonical坐标不是已经匹配米制尺寸的最终场景坐标，插入时需按原规则等比定尺度。其他物体最长稳健范围归一化为2个任意单位，不是2米，真实尺寸和朝向尚未逐一确认。

metadata保留实验时的质量状态与来源记录；车辆canonical.canonical_path已改为包内相对路径（以本包根目录为基准）。其余绝对source/evidence路径是原服务器追溯记录，不是打开PLY所需依赖。质量字段是原生成时快照，不代表全部资产已完成入库或场景插入验收。
本包仅含所要求三条路线的资产输出、元数据和索引；没有场景视频、轨迹、背景checkpoint、原始图片或其他实验路线。
'''
 csvbuf=io.StringIO();w=csv.DictWriter(csvbuf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for path,content in [('README.txt',readme),('INDEX.csv',csvbuf.getvalue()),('INDEX.json',json.dumps(rows,ensure_ascii=False,indent=2))]:
  raw=content.encode('utf-8-sig' if path.endswith('.csv') else 'utf-8');z.writestr(name+'/'+path,raw);hashes.append((hashlib.sha256(raw).hexdigest(),path))
 z.writestr(name+'/SHA256SUMS.txt',''.join(h+'  '+p+'\n' for h,p in hashes))
# Read the actual archive, check all payload hashes and exact requested coverage.
with zipfile.ZipFile(tmp) as z:
 actual=set(z.namelist());assert sum(p.endswith('/canonical.ply') for p in actual)==60
 assert sum(p.endswith('/metadata.json') for p in actual)==60
 for v in variants:
  for group,_,_ in groups:assert len([r for r in rows if r['route']==v and r['group']==group])==10
 for h,p in hashes:
  digest=hashlib.sha256()
  with z.open(name+'/'+p) as f:
   for b in iter(lambda:f.read(8<<20),b''):digest.update(b)
  assert digest.hexdigest()==h,p
 assert len(actual)==124,len(actual)
tmp.replace(archive);h=sha_file(archive);archive.with_suffix('.zip.sha256').write_text(h+'  '+archive.name+'\n')
result=dict(status='PASS',archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=h,asset_count=60,identity_count=20,variants=variants,per_variant=dict(vehicles=10,other_objects=10),archive_members=124,all_packaged_file_hashes_checked=True,ply_bytes_unchanged=True,obsolete_delivery_bytes_removed=sum(x['bytes'] for x in removed))
(R/'docs/assets_only_delivery.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
