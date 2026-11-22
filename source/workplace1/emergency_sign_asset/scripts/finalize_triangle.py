import json,sys,hashlib
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'DriveLab'))
from asset_extraction.gaussians import load_rigid,_export_ply
from asset_extraction.export import inspect_ply
r=W/'runs/V01_B0';d=r/'curation';g=load_rigid(d/'axis.ply');xyz=g.xyz[0]
# The visually reviewed black weighted base is the lowest physical component.
support=(g.opacity>.5)&(g.color.max(1)<.25)&(xyz[:,2]<.12)
assert support.sum()>100
bottom=float(xyz[support,2].min());shift=np.array([0,0,-bottom])
_export_ply(d/'axis.ply',d/'canonical.ply',np.eye(3),shift)
c=json.loads((d/'axis.json').read_text());c['translation']=(np.array(c['translation'])+shift).tolist()
solid=xyz[g.opacity>.5]+shift;bounds=[solid.min(0).tolist(),solid.max(0).tolist()];height=float(solid[:,2].max())
c.update(canonical_path=str(d/'canonical.ply'),sha256=hashlib.sha256((d/'canonical.ply').read_bytes()).hexdigest(),
 heading=dict(front='+X',azimuth_deg=0,confidence=.9,source=str(d/'axis_render/evidence.jpg'),verification=dict(front='A',confidence=.9),note='Primary observed face selected; geometrically completed rear is not source-verified'),heading_verified=True,
 grounding='minimum Gaussian center of visually identified black weighted base, opacity > 0.5; no percentile clipping',
 support_contact=dict(selector='black RGB max < 0.25, z < 0.12, opacity > 0.5 in axis.ply',count=int(support.sum()),old_min_z=bottom,translation=shift.tolist(),min_center_z_after=0.,gaussian_tail_note='Gaussian tails extend below center-defined contact plane'),
 full_opaque_bounds=bounds,scale_reference_height=height,validation=inspect_ply(d/'canonical.ply'))
scale=.42037/height
c['metric_preview']=dict(height_m=.42037,width_m=float(np.ptp(solid[:,1])*scale),depth_m=float(np.ptp(solid[:,0])*scale),isotropic_factor=scale,reference='manufacturer single deployed triangle diagram; depth inferred and unverified')
(d/'canonical.json').write_text(json.dumps(c,indent=2));print(c['metric_preview'])
