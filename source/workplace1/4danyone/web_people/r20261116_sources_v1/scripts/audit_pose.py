"""Every-frame original-pixel projection panels and the maintained temporal check."""
import os,sys,json
import av,numpy as np
from PIL import Image,ImageDraw,ImageOps
from scipy.spatial.transform import Rotation
from safetensors.numpy import load
from prepare_person import ROOT,PROJECT,save,sha
os.environ['HUMAN_RUN_ROOT']=str(ROOT)
sys.path[:0]=[str(ROOT/'code'),str(PROJECT/'repo')]
from frontend import classify_temporal_signals
from fdanyone.skeleton.keypoints import LINKS,KEYPOINT_NAMES
names=('nose','left-eye','right-eye','left-ear','right-ear','left-shoulder','right-shoulder','left-elbow','right-elbow','left-wrist','right-wrist','left-hip','right-hip','left-knee','right-knee','left-ankle','right-ankle')
for aid in sys.argv[1:]:
 d=ROOT/'inputs'/aid;dest=ROOT/'reviews'/aid;p=np.load(d/'research/projection.npz')
 tensors=load((d/'motion_pair/static_estimated/motion.safetensors').read_bytes());obs=p['observed_crop']
 np.testing.assert_allclose(tensors['observed_keypoints_2d'],obs,atol=1e-5,rtol=0)
 distance=np.linalg.norm(p['skeleton_crop_uv'][:,[KEYPOINT_NAMES.index(n) for n in names]]-obs[...,:2],axis=-1)
 height=p['crop_boxes'][:,3]-p['crop_boxes'][:,1];errors=distance/height[:,None]
 rotation=Rotation.from_rotvec(tensors['smpl_params_incam.body_pose'].reshape(-1,3)).as_matrix().reshape(121,21,3,3)
 steps=Rotation.from_matrix((rotation[1:]@rotation[:-1].swapaxes(-1,-2)).reshape(-1,3,3)).magnitude().reshape(120,21)
 steps=np.concatenate((np.zeros((1,21)),steps))
 groups=[([11,13,15],[0,3,6,9]),([12,14,16],[1,4,7,10]),([5,7,9],[12,15,17,19]),([6,8,10],[13,16,18,20])]
 policy=json.loads((ROOT/'TEMPORAL_POLICY.json').read_text())
 result=classify_temporal_signals(np.stack([obs[:,ids,2].mean(1) for ids,_ in groups],axis=1),
 np.stack([steps[:,ids].max(1) for _,ids in groups],axis=1),np.stack([np.quantile(errors[:,ids],.75,axis=1) for ids,_ in groups],axis=1),policy)
 result.update(source_sha256={str(f):sha(f) for f in [d/'research/projection.npz',d/'motion_pair/static_estimated/motion.safetensors']},visual_review_passed=False)
 save(dest/'pose_temporal_check.json',result)
 tiles=[]
 with av.open(str(d/'crop_unsegmented.mp4')) as c:
  for i,f in enumerate(c.decode(video=0)):
   im=f.to_image();draw=ImageDraw.Draw(im);uv=p['skeleton_crop_uv'][i]
   for _,a,b,_,major in LINKS:
    if major:draw.line([tuple(uv[a]),tuple(uv[b])],fill='lime',width=3)
   for x,y,conf in obs[i]:
    if conf>.3:draw.ellipse((x-2,y-2,x+2,y+2),fill='red')
   im=ImageOps.contain(im,(160,300));tile=Image.new('RGB',(160,320),'white');tile.paste(im,(0,20));ImageDraw.Draw(tile).text((3,2),f'{i} / {i/25:.2f}s',fill='black');tiles.append(tile)
 for start in range(0,121,40):
  subset=tiles[start:start+40];sheet=Image.new('RGB',(1280,320*((len(subset)+7)//8)),'#dddddd')
  for i,tile in enumerate(subset):sheet.paste(tile,((i%8)*160,(i//8)*320))
  sheet.save(dest/f'pose_all_{start:03d}.jpg',quality=92)
 print(aid,result['decision'],result.get('reason'),flush=True)
