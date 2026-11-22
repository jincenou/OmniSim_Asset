from pathlib import Path
import json, numpy as np
from PIL import Image,ImageOps,ImageDraw
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
R=Path(__file__).resolve().parents[1];O=R/'nonvehicle10';root=Path('/mnt/public_data/nuScenes')
n=NuScenes(version='v1.0-trainval',dataroot=str(root),verbose=False)
scenes={s['token']:s['name'] for s in n.scene if s['name'] in ['scene-0103','scene-0104','scene-0277','scene-0795']}
obs={}
for s in n.sample:
 if s['scene_token'] not in scenes:continue
 for tok in s['anns']:
  a=n.get('sample_annotation',tok)
  if a['category_name'] not in ['movable_object.trafficcone','movable_object.barrier']:continue
  for cam,st in s['data'].items():
   if not cam.startswith('CAM_'):continue
   sd=n.get('sample_data',st);fp=root/sd['filename']
   if not fp.exists():continue
   cs=n.get('calibrated_sensor',sd['calibrated_sensor_token']);ep=n.get('ego_pose',sd['ego_pose_token']);b=n.get_box(tok);b.translate(-np.array(ep['translation']));b.rotate(Quaternion(ep['rotation']).inverse);b.translate(-np.array(cs['translation']));b.rotate(Quaternion(cs['rotation']).inverse)
   corners=b.corners()
   if corners[2].min()<=.1:continue
   K=np.array(cs['camera_intrinsic']);xy=view_points(corners,K,normalize=True)[:2];lo=xy.min(1);hi=xy.max(1);w,h=hi-lo
   if min(w,h)<25 or max(w,h)<60 or lo.min()<3 or hi[0]>sd['width']-3 or hi[1]>sd['height']-3:continue
   rec=dict(dataset='nuscenes',category='cone' if a['category_name'].endswith('trafficcone') else 'barrier',scene=scenes[s['scene_token']],instance_token=a['instance_token'],annotation_token=tok,sample_data_token=st,image_path=str(fp),image_timestamp_s=sd['timestamp']/1e6,camera=cam,box_xyxy=[*lo.tolist(),*hi.tolist()],area=float(w*h),camera_intrinsic=cs['camera_intrinsic'],camera_to_ego=cs,ego_to_world=ep,object_world=a)
   obs.setdefault((a['instance_token'],cam),[]).append(rec)
ranked=sorted([v for v in obs.values() if len(v)>=4],key=lambda x:max(r['area'] for r in x),reverse=True)
seen=set();selected=[]
for rs in ranked:
 if rs[0]['instance_token'] in seen:continue
 seen.add(rs[0]['instance_token']);r=max(rs,key=lambda x:x['area']);r=dict(r,observations=rs);idx=len(selected);im=Image.open(r['image_path']);x0,y0,x1,y1=r['box_xyxy'];pad=.12*max(x1-x0,y1-y0);crop=[max(0,int(x0-pad)),max(0,int(y0-pad)),min(im.width,int(x1+pad)),min(im.height,int(y1+pad))];fp=O/'inputs'/f'nusc_candidate_{idx:02}.png';im.crop(crop).save(fp);r.update(candidate_id=f'nusc_candidate_{idx:02}',path=str(fp),crop_box=crop);selected.append(r)
 if len(selected)>=24:break
(O/'inputs/nusc_candidates.json').write_text(json.dumps(selected,indent=2));sheet=Image.new('RGB',(1500,300*((len(selected)+5)//6)),'#ddd');d=ImageDraw.Draw(sheet)
for i,r in enumerate(selected):
 xx=i%6*250;yy=i//6*300;im=ImageOps.contain(Image.open(r['path']),(245,250));sheet.paste(im,(xx,yy+40));d.text((xx,yy),r['candidate_id']+' '+r['category'],fill='black');d.text((xx,yy+18),r['scene']+' views='+str(len(r['observations'])),fill='black')
sheet.save(O/'docs/nusc_candidates.jpg');print('DONE',len(selected),flush=True)
