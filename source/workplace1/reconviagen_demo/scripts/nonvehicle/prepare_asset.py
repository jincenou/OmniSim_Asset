from pathlib import Path
import sys,json,argparse
import numpy as np
from plyfile import PlyData,PlyElement
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asset_adapter import inspect,transform,UP_FIX
R=Path(__file__).resolve().parents[2]/'nonvehicle10'
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--variant',required=True);a=p.parse_args();out=R/'results'/a.case/(a.variant+'_seed42');case=json.loads((R/'inputs'/a.case/'manifest.json').read_text());src=out/'raw.ply';v=PlyData.read(str(src))['vertex'].data.copy();bad=np.isinf(v['opacity']);repair=dict(infinite_opacity_count=int(bad.sum()),maximum_activated_opacity_change=1e-7)
if np.isnan(v['opacity']).any():raise ValueError('NaN opacity')
if bad.any():
 v['opacity'][bad]=np.sign(v['opacity'][bad])*np.log((1-1e-7)/1e-7);src=out/'finite.ply';PlyData([PlyElement.describe(v,'vertex')],text=False).write(str(src))
(out/'serialization_repair.json').write_text(json.dumps(repair,indent=2));before,gs=inspect(src);rot=UP_FIX if a.variant.startswith('B0') else np.eye(3);xyz=gs.xyz[0].astype(float)@rot.T;lo,hi=np.quantile(xyz,[.005,.995],axis=0);size=hi-lo
if min(size)<=1e-5:raise ValueError('Degenerate bounds')
# Preserve backend yaw; no source-image view search, and no long-axis rotation for poles/signs.
scale=2./max(size);translation=-np.array([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]])*scale;asset=out/'asset';asset.mkdir(exist_ok=True);after=transform(src,asset/'canonical.ply',rot,translation,scale)
meta=dict(case_id=a.case,category=case['category'],variant=a.variant,payload='native Gaussian PLY; SH0/log-scale/opacity-logit/wxyz',raw_up='-y' if a.variant.startswith('B0') else 'z',canonical_up='z',normalization='robust maximum dimension = 2 arbitrary units; XY centered; base z=0',uniform_scale=scale,rotation=rot.tolist(),translation=translation.tolist(),metric_scale_known=False,heading_verified=False,source_manifest=str(R/'inputs'/a.case/'manifest.json'),quality_status='pending_visual_review',source_validation=before,reloaded_validation=after)
(asset/'metadata.json').write_text(json.dumps(meta,indent=2));print(a.case,a.variant,'EXPORTED',flush=True)
