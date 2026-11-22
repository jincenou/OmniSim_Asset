from pathlib import Path
import json,os
import numpy as np
ROOT=Path(os.environ['FDANYONE_WAYMO_ROOT']);OLD=Path(os.environ['FDANYONE_BASELINE'])
new=json.loads((ROOT/'preflight/conditioning/cameras.json').read_text())
old=json.loads((OLD/'cameras.json').read_text())
assert len(new['cameras'])==len(old['cameras'])==24
errors=[]
for a,b in zip(new['cameras'],old['cameras'],strict=True):
 assert a['camera_id']==b['camera_id']
 errors.extend(float(np.abs(np.asarray(a[k])-np.asarray(b[k])).max()) for k in ['K','camera_to_world'])
 # Compact delivered cameras store only camera_to_world; compare the explicit inverse.
 errors.append(float(np.abs(np.asarray(a['world_to_camera'])-np.linalg.inv(np.asarray(b['camera_to_world']))).max()))
assert max(errors)<1e-6,errors
a=np.load(ROOT/'preflight/root_motion.npz');b=np.load(OLD/'root_motion.npz')
root_error=max(float(np.abs(a[k]-b[k]).max()) for k in b.files)
assert root_error<1e-6,root_error
report={'state':'camera_and_root_reproduction_verified','camera_max_abs':max(errors),'root_max_abs':root_error,
 'original_input_and_motion_reused':True,'source_crop':'Recomputed by unchanged pinned BiRefNet and source_crop logic; original numeric crop was not retained, so historical crop equivalence not asserted.'}
(ROOT/'research/baseline_pairing.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
