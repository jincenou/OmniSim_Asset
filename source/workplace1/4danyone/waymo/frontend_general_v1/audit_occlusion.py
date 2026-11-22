from pathlib import Path
import json,hashlib
import torch,numpy as np
from PIL import Image
G=Path(__file__).resolve().parent;O=G/'occlusion_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[]
for job in json.loads((O/'jobs.json').read_text()):
 root=Path(job['root']);meta=json.loads((root.parent/'case.json').read_text());src=Path(meta['original_source']);cached=src/'motion_pair/runtime/calibrated/preprocess';fresh=root/'motion_pair/runtime/calibrated/preprocess';oldfeat=torch.load(cached/'vit_features.pt',map_location='cpu',weights_only=True);newfeat=torch.load(fresh/'vit_features.pt',map_location='cpu',weights_only=True)
 exact=torch.equal(oldfeat,newfeat)
 if job['branch']=='kp_only':assert exact
 else:assert not exact
 kp=torch.load(fresh/'vitpose.pt',map_location='cpu',weights_only=True);raw=torch.load(root/'motion_pair/detector_before_known_occluder.pt',map_location='cpu',weights_only=True);occ=np.load(root.parent/'evidence_masks.npz')['occluders'];xy=np.rint(raw[...,:2].numpy()).astype(int);h,w=occ.shape[1:]
 inside=(xy[...,0]>=0)&(xy[...,0]<w)&(xy[...,1]>=0)&(xy[...,1]<h);hidden=inside&(occ[np.arange(len(occ))[:,None],xy[...,1].clip(0,h-1),xy[...,0].clip(0,w-1)]>0)
 assert torch.equal(kp[~hidden],raw[~hidden]) and (kp[hidden]==0).all()
 if job['branch']=='rgb':
  native=json.loads((root/'native.json').read_text());old=json.loads((src/'native.json').read_text());mask=np.load(root.parent/'evidence_masks.npz');rect=mask['native_rectangles'];red=mask['native_redacted']
  for i,(a,b) in enumerate(zip(native['frames'],old['frames'],strict=True)):
   x=np.array(Image.open(a['path']).convert('RGB'));y=np.array(Image.open(b['path']).convert('RGB'));assert sha(a['path'])==a['sha256']
   if red[i]:
    x0,y0,x1,y1=rect[i];assert np.all(x[y0:y1,x0:x1]==128);y[y0:y1,x0:x1]=128
   np.testing.assert_array_equal(x,y)
  assert sha(root/'native_cameras.npz')==sha(src/'native_cameras.npz')
 for key in ['K_input','R_world_to_cv_camera','source_relative_exposure_seconds']:
  np.testing.assert_array_equal(np.load(root/'inputs/cameras.npz')[key],np.load(src/'inputs/cameras.npz')[key])
 rows.append({'job':job['key'],'image_features_equal_unoccluded':exact,'feature_mae_to_unoccluded':float((oldfeat-newfeat).abs().mean()),'known_occluded_keypoints_zeroed':int(hidden.sum()),'unmasked_keypoints_preserved':True,'camera_and_times_unchanged':True})
(O/'input_isolation_audit.json').write_text(json.dumps({'state':'passed','jobs':rows,'scope':'Tensor equality, pixel edits and input-camera checks; source code also restricts reference reads to evaluation scripts. Keypoint-only branch intentionally retains unoccluded visual evidence.'},indent=2)+'\n')
print(json.dumps({'jobs':len(rows),'passed':True}))
