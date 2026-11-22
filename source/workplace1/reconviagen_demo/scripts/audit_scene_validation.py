from pathlib import Path
import json,hashlib,csv,numpy as np
from PIL import Image
import cv2
R=Path(__file__).resolve().parents[1];O=R/'scene_validation';variants=['B0','R','H','B0_real4'];rows=[];checks=[]
for scene in ['scene-0103','scene-0104','scene-0277']:
 bg=O/'results'/scene/'random10_04_white_suv/background'
 for cid in ['random10_04_white_suv','random10_02_white_pickup']:
  base=O/'results'/scene/cid;metas=[json.loads((base/v/'meta.json').read_text()) for v in variants];contacts=[json.loads((base/v/'contact_trajectory.json').read_text()) for v in variants];calibs=[json.loads((base/v/'color_calibration.json').read_text()) for v in variants]
  assert len({m['trajectory_sha256'] for m in metas})==1 and all(np.array_equal(contacts[0],x) for x in contacts[1:]);assert all(c==calibs[0] for c in calibs[1:]);assert all(m['n_frames']==60 and m['start_frame']==30 and m['fps']==12 for m in metas)
  for v,meta in zip(variants,metas):
   d=json.loads((base/v/'visibility_diagnostics.json').read_text());assert len(d)==360;visible=[r for r in d if r['isolated_pixels_01']>=64];nf=sum(any(r['frame']==f and r['visible_pixels_01']>=64 for r in d) for f in range(60));front=[r for r in d if r['sensor']=='FRONT'];mass=sum(r['visible_alpha_mass'] for r in visible)/sum(r['isolated_alpha_mass'] for r in visible);edge=sum(r['edge_clipped'] for r in visible);support=np.array(contacts[0]);tilts=np.degrees(np.arccos(support[:,4]));errors=[]
   # Rear-camera tiles are unchanged when the inserted object is invisible there.
   for f in [0,30,59]:
    a=np.array(Image.open(base/v/'frames'/f'f{f:03}.png')).astype(float);b=np.array(Image.open(bg/'frames'/f'f{f:03}.png')).astype(float);assert a.shape==b.shape
    for j,cam in enumerate(['BACK_LEFT','BACK','BACK_RIGHT']):
     row=next(r for r in d if r['frame']==f and r['sensor']==cam)
     if row['isolated_alpha_mass']<1e-4:errors.append(float(np.abs(a[450:900,j*800:(j+1)*800]-b[450:900,j*800:(j+1)*800]).max()))
   assert max(errors,default=0)==0,'Background changed outside asset'
   rec=dict(scene=scene,case_id=cid,variant=v,frames=60,camera_renders=360,frames_with_visible_asset=nf,occlusion_visible_alpha_fraction=float(mass),clipped_camera_frames=edge,projected_camera_frames=len(visible),mean_front_foreground_pixels=float(np.mean([r['visible_pixels_01'] for r in front])),mean_trajectory_tilt_deg=float(tilts.mean()),background_unaffected_tiles_max_abs_error=max(errors,default=0),gaussian_count=meta['gaussian_count'],render_seconds=meta['elapsed_seconds']);rows.append(rec)
  checks.append(dict(scene=scene,case_id=cid,shared_contact_exact=True,shared_trajectory=True,shared_color_probe_exact=True,shared_camera_indices=True,unchanged_background_tiles_exact=True))
result=dict(status='PASS',insertions=len(rows),frames=sum(r['frames'] for r in rows),camera_renders=sum(r['camera_renders'] for r in rows),rows=rows,controlled_comparison_checks=checks,metric_cautions=['Visible alpha fraction measures actual occlusion, not quality; lower is not inherently worse','Scene output success is not perceptual integration quality','No PSNR/SSIM against empty background: there is no ground-truth image containing the inserted actor'])
(O/'docs/insertion_audit.json').write_text(json.dumps(result,indent=2))
with (O/'docs/insertion_diagnostics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('PASS',len(rows),'insertions',result['camera_renders'],'camera renders',flush=True)
