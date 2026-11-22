from pathlib import Path
import sys,json,re,hashlib,zipfile,time
R=Path(__file__).resolve().parents[1];O=R/'deliveries/vehicle_experiments_20261023';sys.path.insert(0,str(R/'tools/report_deps'))
from docx import Document
from PIL import Image
import imageio.v2 as imageio
errors=[];decoded=[]
for p in O.glob('*.html'):
 for ref in re.findall(r'(?:src|href)="([^"]+)"',p.read_text()):
  if not ref.startswith(('http:','https:','#')) and not (p.parent/ref.split('#')[0]).is_file():errors.append('Missing '+ref)
for p in (O/'images').rglob('*'):
 if p.is_file():
  with Image.open(p) as im:im.verify()
for p in sorted((O/'videos').glob('*.mp4')):
 reader=imageio.get_reader(p);meta=reader.get_meta_data();n=sum(1 for f in reader);reader.close();assert n==96,(p,n)
 decoded.append(dict(file=str(p.relative_to(O)),decoded_frames=n,fps=meta['fps'],size=list(meta['size'])))
 print(p.name,n,'PASS',flush=True)
assert len(decoded)==13
cases=json.loads((R/'inputs/random10_cases.json').read_text())
for c in cases:
 tag=c['case_id'].replace('random10_','')
 for original,copy in [(R/'results'/c['case_id']/'comparison_matched_source.jpg',O/'images/comparisons'/f'{tag}_matched.jpg'),(R/'results'/c['case_id']/'comparison_six_routes.mp4',O/'videos'/f'{tag}_six_routes.mp4')]:assert hashlib.sha256(original.read_bytes()).digest()==hashlib.sha256(copy.read_bytes()).digest()
doc=Document(O/'汇报正文.docx');assert len(doc.tables)==7,len(doc.tables);assert len(doc.inline_shapes)==3,len(doc.inline_shapes)
assert not errors,errors
metrics=json.loads((O/'metrics/random10_pixel_metrics.json').read_text());assert len(metrics['rows'])==140;assert len(metrics['summary'])==12
for mode in ['normalized','native']:
 for v in ['B0','R','F','H','B0_real1','B0_real4']:
  rows=[x for x in metrics['rows'] if x['alignment']==mode and x['variant']==v and x['seed']==42];assert len(rows)==10
  summary=next(x for x in metrics['summary'] if x['alignment']==mode and x['variant']==v)
  for k in ['PSNR_dB','SSIM','LPIPS_alex']:
   assert abs(sum(x[k] for x in rows)/10-summary[k]['mean'])<1e-10
report=dict(status='PASS',checked_unix=time.time(),n_videos=13,decoded_frames=sum(x['decoded_frames'] for x in decoded),videos=decoded,original_main_images_and_videos_hash_match=True,full_frame_to_frozen_crop_pixel_match='Verified during package_report_media.py extraction for all10',docx_tables=len(doc.tables),docx_embedded_figures=len(doc.inline_shapes),html_local_links_valid=True,pixel_metric_rows=140,n_cases=10,n_routes=6,n_assets=70)
(O/'reproducibility/package_audit.json').write_text(json.dumps(report,indent=2))
lines=[]
for p in sorted(O.rglob('*')):
 if p.is_file() and p.name!='SHA256SUMS.txt':lines.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(O)))
(O/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n')
archive=O.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(O.rglob('*')):
  if p.is_file():z.write(p,arcname=str(p.relative_to(O.parent)))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 # Verify actual archived bytes match every manifest digest.
 for line in lines:
  digest,name=line.split('  ',1);assert hashlib.sha256(z.read(O.name+'/'+name)).hexdigest()==digest
size=archive.stat().st_size;assert size<60*1024**2
result=dict(**report,archive=str(archive),archive_bytes=size,archive_MiB=size/1024**2,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(lines)+1)
(R/'docs/advisor_package_delivery.json').write_text(json.dumps(result,indent=2));archive.with_suffix('.zip.sha256').write_text(result['sha256']+'  '+archive.name+'\n')
print('ZIP VERIFIED',size/1024**2,'MiB',str(archive),flush=True)
