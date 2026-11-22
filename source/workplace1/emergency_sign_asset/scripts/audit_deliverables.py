import json,hashlib,os
from pathlib import Path
import numpy as np
from plyfile import PlyData
W=Path(__file__).resolve().parents[1];B=Path('/mnt/ojc/workplace1/asset_library')
rows=[r for r in json.loads((B/'manifests/asset_library_rigid.json').read_text())['assets'] if r.get('dataset')=='emergency_traffic_signs'];out=[]
for r in rows:
 p=B/r['package_path'];receipt=json.loads((p/'package.json').read_text());assert all(hashlib.sha256((p/f).read_bytes()).hexdigest()==sha for f,sha in receipt['files'].items());v=PlyData.read(str(p/'asset.ply'))['vertex'].data;xyz=np.column_stack([v[k] for k in ['x','y','z']]);op=1/(1+np.exp(-v['opacity']));solid=xyz[op>.5];c=r['canonical'];m=c['metric_preview'];fac=m['height_m']/c['scale_reference_height'];assert np.isclose(fac,m['isotropic_factor']);assert abs(solid[:,2].min()*fac)<.003
 q=np.column_stack([v[f'rot_{i}'] for i in range(4)]);assert np.allclose(np.linalg.norm(q,axis=1),1,atol=1e-5)
 assert r['category']=='traffic_sign' and c['heading_verified'] and c['heading']['front']=='+X'
 assert all((p/f).exists() for f in ['asset.ply','asset.json','source_camera.json','placement.json','package.json','sign_semantics.json','quality.json','provenance.json','refs/source_rgba.png','previews/orbit.mp4'])
 out.append(dict(uid=r['uid'],status=r['status'],package=str(p),sha256=c['sha256'],gaussians=len(v),actual_height_m=float(solid[:,2].max()*fac),target_height_m=m['height_m'],min_support_center_m=float(solid[:,2].min()*fac),metric_scale_once=fac,all_package_hashes_valid=True))
comparisons=[]
for sid in ['V01','V05','P01']:
 rr=[]
 for suffix in ['B0','H0']:
  g=json.loads((W/'runs'/f'{sid}_{suffix}'/'generation/generation.json').read_text());row=g['assets'][0];review=json.loads((W/'runs'/f'{sid}_{suffix}'/'generation/asset_00000/flux_review.json').read_text());rr.append(dict(run=f'{sid}_{suffix}',flux_sha256=row['stages']['flux']['images'][0]['sha256'],rgba_sha256=review['images'][0]['rgba_sha256'],seed=g['config']['seed'],k=len(row['real_images'])))
 assert rr[0]['flux_sha256']==rr[1]['flux_sha256'] and rr[0]['rgba_sha256']==rr[1]['rgba_sha256'] and rr[0]['seed']==rr[1]['seed'] and all(r['k']==0 for r in rr);comparisons.append(rr)
report=dict(packages=out,matched_comparisons=comparisons,notes=['No default keep admission for review assets','Normalized PLY and one isotropic target-height scale; Gaussian tails may extend below center-defined contact plane'])
(W/'reports/delivery_validation.json').write_text(json.dumps(report,indent=2));print('Validated',len(out),'packages and',len(comparisons),'matched B0/H0 comparisons')
