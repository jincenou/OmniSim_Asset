from pathlib import Path
import json,sys,csv,hashlib,re,zipfile,time,gc
import numpy as np
from PIL import Image
import imageio.v2 as imageio
from concurrent.futures import ThreadPoolExecutor
R=Path(__file__).resolve().parents[1];N=R/'nonvehicle10';S=R/'scene_validation';D=R/'deliveries/nonvehicle10_and_scene_validation_20261023';sys.path.insert(0,str(R/'tools/report_deps'));from docx import Document
from asset_adapter import inspect
cases=json.loads((N/'inputs/cases.json').read_text());export_checks=[]
for c in cases:
 cid=c['case_id'];base=json.loads((N/'inputs'/cid/'segmented_manifest.json').read_text());real={r['id'] for r in base['images'] if r['source_type']=='real'};flux={r['id'] for r in base['images'] if r['source_type']=='flux'};assert len(flux)==1
 for v in ['B0','R','H','B0_real1']:
  out=N/'results'/cid/f'{v}_seed42';inp=json.loads((out/'input_manifest.json').read_text());ids={r['id'] for r in inp['images']};expected={'B0':flux,'R':real,'H':real|flux,'B0_real1':flux|{base['flux_reference_id']}}[v];assert ids==expected,(cid,v,ids,expected)
  payload=out/'asset/canonical.ply';audit,gs=inspect(payload);del gs;gc.collect();areas=[]
  for i in range(16):
   p=out/f'orbit/view_{i:02}.png';a=out/f'orbit/alpha_{i:02}.png';Image.open(p).verify();mask=np.array(Image.open(a));assert mask.shape==(512,512);areas.append(int((mask>127).sum()))
  assert min(areas)>0
  for enc in ['clip','dino']:
   m=json.loads((out/f'{enc}_metrics.json').read_text());assert len(m['view_cosines'])==16 and abs(m['I16']-np.mean(m['view_cosines']))<1e-6
  pix=json.loads((out/'pixel_metrics.json').read_text());assert len(pix['view_scores'])==16
  for k,value in pix['mean'].items():assert abs(value-np.mean([r[k] for r in pix['view_scores']]))<1e-8
  record=dict(case_id=cid,variant=v,ply_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),valid_original_loader=True,gaussian_count=audit['gaussian_count'],min_foreground_pixels=min(areas),frozen_inputs_verified=True,metrics_complete=True);export_checks.append(record);(out/'completion.json').write_text(json.dumps(record,indent=2))
 print('ASSETS VERIFIED',cid,flush=True)
metric=json.loads((N/'docs/metrics.json').read_text());assert len(metric['rows'])==40 and len(metric['common_matched_ids'])==10
for s in metric['summary']:
 rows=[r for r in metric['rows'] if r['variant']==s['variant']]
 for k in ['CLIP_I16','DINO_I16','PSNR_I16','SSIM_I16','LPIPS_I16']:assert abs(s[k]['mean']-np.mean([r[k] for r in rows]))<1e-8
assert json.loads((S/'docs/insertion_audit.json').read_text())['status']=='PASS'
for p in D.rglob('*.html'):
 for ref in re.findall(r'(?:src|href)="([^"]+)"',p.read_text()):
  if not ref.startswith(('http:','https:','#')):assert (p.parent/ref.split('#')[0]).is_file(),(p,ref)
for p in D.rglob('*'):
 if p.suffix.lower() in ['.jpg','.png']:Image.open(p).verify()
for sub,nt in [('nonvehicle10',5),('scene_validation',3)]:
 doc=Document(D/sub/'REPORT.docx');assert len(doc.tables)==nt;assert len(doc.inline_shapes)==3
videos=list(D.rglob('*.mp4'));assert len(videos)==19
def decode(p):
 reader=imageio.get_reader(p);meta=reader.get_meta_data();count=sum(1 for _ in reader);reader.close();expect=96 if 'nonvehicle10/results' in str(p.relative_to(D)) else 60;assert count==expect,(p,count);return dict(file=str(p.relative_to(D)),frames=count,fps=meta['fps'],resolution=list(meta['size']))
with ThreadPoolExecutor(max_workers=4) as pool:decoded=list(pool.map(decode,videos))
report=dict(status='PASS',checked_unix=time.time(),gaussian_assets=40,scene_insertions=24,scene_camera_renders=8640,display_videos=19,decoded_pack_frames=sum(x['frames'] for x in decoded),videos=decoded,export_checks=export_checks,html_local_links_valid=True,docx_tables_and_figures_valid=True,metric_means_recomputed=True,nonvehicle_source_input_sets_verified=True,nonvehicle_visual_quality='Manual review retained; AOR success does not imply semantic correctness',scene_visual_quality='Shared conditions audited; floating impression/no contact shadows/color mismatch remain')
(D/'reproducibility/delivery_audit.json').write_text(json.dumps(report,indent=2));(N/'docs/completion_audit.json').write_text(json.dumps(report,indent=2))
lines=[]
for p in sorted(D.rglob('*')):
 if p.is_file() and p.name!='SHA256SUMS.txt':lines.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(D)))
(D/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n');archive=D.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(D.rglob('*')):
  if p.is_file():z.write(p,arcname=str(p.relative_to(D.parent)))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 for line in lines:
  digest,name=line.split('  ',1);assert hashlib.sha256(z.read(D.name+'/'+name)).hexdigest()==digest
size=archive.stat().st_size;assert size<50*1024**2
result=dict(status='PASS',archive=str(archive),bytes=size,MiB=size/1024**2,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),gaussian_assets=40,scene_insertions=24,pack_videos=19,files=len(lines)+1)
(R/'docs/nonvehicle_scene_delivery.json').write_text(json.dumps(result,indent=2));archive.with_suffix('.zip.sha256').write_text(result['sha256']+'  '+archive.name+'\n');print(json.dumps(result,indent=2),flush=True)
