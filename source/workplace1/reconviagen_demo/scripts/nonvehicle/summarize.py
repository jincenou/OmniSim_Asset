from pathlib import Path
import json,csv,numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[2]/'nonvehicle10';cases=json.loads((R/'inputs/cases.json').read_text());variants=['B0','R','H','B0_real1'];rows=[]
for c in cases:
 for v in variants:
  out=R/'results'/c['case_id']/f'{v}_seed42';rec=dict(case_id=c['case_id'],category=c['category'],variant=v,real_available=len(c['images']),valid=False,failure=None)
  try:
   meta=json.loads((out/'asset/metadata.json').read_text());render=json.loads((out/'orbit/render_config.json').read_text());areas=[int((np.array(Image.open(out/f'orbit/alpha_{i:02}.png'))>127).sum()) for i in range(16)];assert min(areas)>0;rec.update(valid=True,gaussian_count=meta['reloaded_validation']['gaussian_count'],min_orbit_foreground_pixels=min(areas),max_orbit_foreground_pixels=max(areas))
   for enc in ['clip','dino']:rec[enc.upper()+'_I16']=json.loads((out/f'{enc}_metrics.json').read_text())['I16']
   pix=json.loads((out/'pixel_metrics.json').read_text())['mean'];rec.update(PSNR_I16=pix['PSNR_dB'],SSIM_I16=pix['SSIM'],LPIPS_I16=pix['LPIPS_alex'],silhouette_IoU_I16=pix['silhouette_IoU'],PSNR_union_I16=pix['PSNR_union_dB'],SSIM_union_I16=pix['SSIM_union'])
  except Exception as ex:rec['failure']=str(ex)
  rows.append(rec)
common=[c['case_id'] for c in cases if all(next(r for r in rows if r['case_id']==c['case_id'] and r['variant']==v)['valid'] and 'LPIPS_I16' in next(r for r in rows if r['case_id']==c['case_id'] and r['variant']==v) for v in variants)]
keys=['CLIP_I16','DINO_I16','PSNR_I16','SSIM_I16','LPIPS_I16'];summary=[];categories=[]
for v in variants:
 sel=[r for r in rows if r['variant']==v and r['case_id'] in common];s=dict(variant=v,Nquery=10,Nvalid=sum(r['valid'] for r in rows if r['variant']==v),matched_n=len(sel));s['AOR']=s['Nvalid']/10
 for k in keys:
  x=np.array([r[k] for r in sel]);rng=np.random.default_rng(20261023);boot=x[rng.integers(0,len(x),(10000,len(x)))].mean(1) if len(x) else np.array([np.nan]);s[k]=dict(mean=float(x.mean()) if len(x) else None,std=float(x.std(ddof=1)) if len(x)>1 else None,bootstrap95=np.quantile(boot,[.025,.975]).tolist())
 summary.append(s)
 for cat in sorted({c['category'] for c in cases}):
  rs=[r for r in sel if r['category']==cat];categories.append(dict(variant=v,category=cat,n=len(rs),**{k:float(np.mean([r[k] for r in rs])) if rs else None for k in keys}))
fid_path=R/'docs/fid_metrics.json'
fids={r['variant']:r for r in json.loads(fid_path.read_text())['results']} if fid_path.exists() else {}
if fids:
 assert len(common)==10
 for ss in summary:ss['FID_2048']=fids[ss['variant']]['FID_2048'];ss['FID_reference_n']=10;ss['FID_rendered_n']=160
result=dict(rows=rows,summary=summary,category_summary=categories,common_matched_ids=common,ED_R=None,ED_R_reason='No independent generated-object source-pose alignment; no best-view substitution')
(R/'docs/metrics.json').write_text(json.dumps(result,indent=2))
for name,data in [('per_asset_metrics',rows),('per_category_metrics',categories)]:
 fields=list(dict.fromkeys(k for r in data for k in r))
 with (R/f'docs/{name}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
flat=[dict(variant=s['variant'],AOR=s['AOR'],n=s['matched_n'],**{k:s[k]['mean'] for k in keys}) for s in summary]
if fids:
 for row in flat:row.update(FID_2048=fids[row['variant']]['FID_2048'],FID_reference_n=10,FID_rendered_n=160)
with (R/'docs/summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
print(json.dumps(flat,indent=2));print('COMMON',len(common))
