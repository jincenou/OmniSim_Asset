from pathlib import Path
import json,argparse,numpy as np,csv,hashlib
from PIL import Image
from evaluate_asset_metrics import Encoders,foreground
R=Path(__file__).resolve().parents[1];O=R/'scene_validation';p=argparse.ArgumentParser();p.add_argument('--encoder',choices=['clip','dino'],required=True);a=p.parse_args();model=Encoders(a.encoder);variants=['B0','R','H','B0_real4'];rows=[]
for cid in ['random10_04_white_suv','random10_02_white_pickup']:
 camera=json.loads((R/'inputs'/cid/'evaluation/camera.json').read_text());source,sm,_=foreground(camera['source_rgb'],camera['source_mask']);sf=model.encode([source],[sm])['global_embedding'][0]
 for scene in ['scene-0103','scene-0104','scene-0277']:
  base=O/'results'/scene/cid;keys=[]
  for frame in [0,30,59]:
   for cam in ['FRONT','FRONT_LEFT','FRONT_RIGHT','BACK','BACK_LEFT','BACK_RIGHT']:
    paths=[base/v/'diagnostics'/f'f{frame:03}_{cam}_visible.png' for v in variants]
    if all(p.exists() and (np.array(Image.open(p))>127).sum()>=64 for p in paths):keys.append((frame,cam))
  if not keys:raise ValueError('No common visible keyframes '+scene+cid)
  for v in variants:
   vals=[];crops=[]
   for frame,cam in keys:
    folder=base/v/'diagnostics';im,mask,crop=foreground(folder/f'f{frame:03}_{cam}.jpg',folder/f'f{frame:03}_{cam}_visible.png');ff=model.encode([im],[mask])['global_embedding'][0];vals.append(float(ff@sf));crops.append(crop)
   row=dict(scene=scene,case_id=cid,variant=v,encoder=a.encoder,source_to_inserted_foreground_similarity=float(np.mean(vals)),common_camera_frame_keys=keys,view_scores=vals,crops=crops,scope='Three fixed timestamps; cameras visible in all four routes; masked scene RGB including actual occlusion/decoder; source appearance diagnostic, not insertion ground-truth or ED-R');rows.append(row);print(scene,cid,v,a.encoder,row['source_to_inserted_foreground_similarity'],flush=True)
(O/'docs'/f'{a.encoder}_scene_metrics.json').write_text(json.dumps(rows,indent=2))
