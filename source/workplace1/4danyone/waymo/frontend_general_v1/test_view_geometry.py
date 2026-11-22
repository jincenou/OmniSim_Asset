"""Asymmetric triangulated debug mesh through actual stored target cameras.
Not a generated-human test. No debug colors are fed to frozen generation weights.
"""
from pathlib import Path
import sys,json,dataclasses,hashlib
import numpy as np
from PIL import Image,ImageDraw
G=Path(__file__).resolve().parent;W=G.parent;REPO=W.parent/'repo';sys.path.insert(0,str(REPO))
from fdanyone.geometry.cameras import Camera,project_points,look_at_pytorch3d,pytorch3d_to_opencv,homogeneous_w2c
from camera_geometry import transform_points
COLORS=np.array([[220,45,45],[40,80,225],[40,200,70],[230,200,30],[220,50,220],[80,190,200]],dtype=np.uint8)
NAMES=['FRONT +z','BACK -z','LEFT +x','RIGHT -x','TOP +y','BOTTOM -y']
def box(lo,hi):
 v=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
 quads=[[1,3,7,5],[0,4,6,2],[4,5,7,6],[0,2,3,1],[2,6,7,3],[0,1,5,4]]
 f=np.array([[q[0],q[1],q[2]] for q in quads]+[[q[0],q[2],q[3]] for q in quads]);return v[f],np.tile(np.arange(6),2)
def render(triangles,labels,K,C,h,w):
 y,x=np.mgrid[:h,:w];pixel=np.stack([x+.5,y+.5,np.ones_like(x)],-1).reshape(-1,3);ray=pixel@np.linalg.inv(K).T@C[:3,:3].T;origin=C[:3,3]
 z=np.full(len(ray),np.inf);ids=np.full(len(ray),-1,int)
 for tri,label in zip(triangles,labels,strict=True):
  a,b,c=tri;e1=b-a;e2=c-a;p=np.cross(ray,e2);det=p@e1;good=np.abs(det)>1e-10;inv=np.zeros_like(det);inv[good]=1/det[good];tvec=origin-a;u=(p@tvec)*inv;q=np.cross(tvec,e1);vv=(ray@q)*inv;distance=(e2@q)*inv
  valid=good&(u>=0)&(vv>=0)&(u+vv<=1)&(distance>0)&(distance<z);z[valid]=distance[valid];ids[valid]=label
 im=np.full((len(ray),3),35,np.uint8);hit=ids>=0;im[hit]=COLORS[ids[hit]];return im.reshape(h,w,3),ids.reshape(h,w)
def main():
 out=G/'geometry_debug';out.mkdir(exist_ok=True);records=[]
 for sid,scene in [('P08','038'),('P36','061')]:
  path=W/'scene20/r20261019/scenes'/scene/sid/'cameras.json';payload=json.loads(path.read_text());cams=payload['cameras'];assert [x['camera_id'] for x in cams]==list(range(24))
  center=np.mean([np.array(r['camera_to_world'])[:3,3] for r in cams],axis=0);cx,cz=center[[0,2]]
  pieces=[box([cx-.22,.5,cz-.13],[cx+.22,1.45,cz+.13]),box([cx-.13,1.45,cz-.13],[cx+.13,1.8,cz+.13]),box([cx-.035,1.60,cz+.13],[cx+.035,1.67,cz+.23])]
  triangles=np.concatenate([a for a,b in pieces]);labels=np.concatenate([b for a,b in pieces]);panels=[];allrows=[]
  for r in cams:
   C=np.array(r['camera_to_world']);K=np.array(r['K']);H=r['image_height'];width=r['image_width'];h=320;w=176;Ksmall=K.copy();Ksmall[0]*=w/width;Ksmall[1]*=h/H
   im,ids=render(triangles,labels,Ksmall,C,h,w);counts=np.bincount(ids[ids>=0],minlength=6)
   assert counts.sum()>100
   # Independent homogeneous projection versus shipped float32 projection routine.
   points=triangles.reshape(-1,3);camera=Camera(r['camera_id'],r['layer_index'],r['yaw'],0,r['pitch'],tuple(C[:3,3]),tuple(map(tuple,K)),tuple(map(tuple,np.linalg.inv(C))),tuple(map(tuple,C)),width,H)
   official,depth,valid=project_points(points,camera);pc=np.c_[points,np.ones(len(points))]@np.linalg.inv(C)[:3].T;uvh=pc@K.T;uv=uvh[:,:2]/uvh[:,2:]
   projection_error=float(np.max(np.abs(uv-official)));assert projection_error<=1e-3
   reconstructed=(np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T)*pc[:,2:];world=np.c_[reconstructed,np.ones(len(reconstructed))]@C[:3].T
   roundtrip=float(np.max(np.abs(world-points)));assert roundtrip<=1e-8
   row={'camera_id':r['camera_id'],'face_pixel_counts':dict(zip(NAMES,counts.tolist(),strict=True)),'projection_max_error_px':projection_error,'world_roundtrip_error':roundtrip};allrows.append(row)
   if r['camera_id'] in [0,4,8,12,16,20]:
    panel=Image.new('RGB',(w,h+26),'#222222');panel.paste(Image.fromarray(im),(0,26));ImageDraw.Draw(panel).text((3,5),f'{sid} view{r["camera_id"]:02d}',fill='white');panels.append(panel)
  assert allrows[0]['face_pixel_counts']['FRONT +z']>100 and allrows[0]['face_pixel_counts']['BACK -z']==0
  assert allrows[12]['face_pixel_counts']['BACK -z']>100 and allrows[12]['face_pixel_counts']['FRONT +z']==0
  sheet=Image.new('RGB',(176*6,346));[sheet.paste(im,(i*176,0)) for i,im in enumerate(panels)];sheet.save(out/f'{sid}_asymmetric_mesh.png')
  records.append({'sample':sid,'camera_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'views':allrows,'front_back_camera_index_test_passed':True})
 # Actual measured moving cameras with independently specified body path/heading.
 c=np.load(W/'frontend_quality_20261020/motion_unmasked/P08/measured_camera_maps.npz')['camera_to_waymo_world'];n=len(c);ts=np.linspace(0,2.4,n);center=c[0,:3,3]+c[0,:3,:3]@np.array([0,0,30]);right=c[0,:3,0];forward=c[0,:3,2];up=-c[0,:3,1]
 template=np.array([[-.2,0,0],[.2,0,0],[-.1,.5,0],[.1,.5,0],[0,.9,.1]])
 motion_tests=[]
 for name,velocity,yaws in [('stationary',np.zeros(3),np.zeros(n)),('walk_under_measured_camera',-forward*.4,np.zeros(n)),('true_backward',forward*.4,np.zeros(n)),('true_side_step',right*.4,np.zeros(n)),('turn_in_place',np.zeros(3),np.linspace(0,np.pi,n))]:
  basis=np.stack([right,up,-forward],1);world=[]
  for t,yaw in zip(ts,yaws,strict=True):
   yawR=np.array([[np.cos(yaw),0,np.sin(yaw)],[0,1,0],[-np.sin(yaw),0,np.cos(yaw)]]);world.append(template@yawR.T@basis.T+center+t*velocity)
  world=np.array(world);cam=transform_points(world,np.linalg.inv(c));assert np.all(cam[...,2]>0)
  recovered=transform_points(cam,c);err=float(np.max(np.abs(recovered-world)));assert err<1e-6
  # Reprojection in a representative calibrated intrinsic. No assumed forward-motion relabeling.
  K=np.array([[1500,0,960],[0,1500,640],[0,0,1.]])
  uvh=cam@K.T;uv=uvh[...,:2]/uvh[...,2:];rec_cam=transform_points(recovered,np.linalg.inv(c));h2=rec_cam@K.T;uv2=h2[...,:2]/h2[...,2:];pe=float(np.max(np.abs(uv2-uv)));assert pe<=1e-3
  facing=-forward
  direction_dot=float(velocity@facing)
  if name=='walk_under_measured_camera':assert direction_dot>0
  if name=='true_backward':assert direction_dot<0
  if name=='true_side_step':assert abs(direction_dot)<1e-8
  motion_tests.append({'velocity_dot_body_facing':direction_dot,'case':name,'world_roundtrip_max_m':err,'reprojection_roundtrip_max_px':pe,'known_heading_is_independent_of_velocity':True})
 # Explicit overtaking geometry: fixed heading +z, walking +z at0.4m/s,
 # camera side-offset3m and +z at2m/s crosses from behind to ahead.
 elapsed=np.linspace(0,2.4,61);root=np.stack([elapsed*0,elapsed*0,elapsed*.4],1);camera_positions=np.stack([elapsed*0+3,elapsed*0+1.2,-2+elapsed*2],1)
 relative_z=camera_positions[:,2]-root[:,2];assert relative_z[0]<0 and relative_z[-1]>0
 max_error=0.
 for body,camera in zip(root,camera_positions,strict=True):
  r,t=look_at_pytorch3d(camera,body+np.array([0,.8,0]));r,t=pytorch3d_to_opencv(r,t);M=homogeneous_w2c(r,t);points=template+body
  local=np.c_[points,np.ones(len(points))]@M[:3].T;assert np.all(local[:,2]>0)
  image=local@K.T;uv=image[:,:2]/image[:,2:];recover_camera=(np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T)*local[:,2:];recover_world=np.c_[recover_camera,np.ones(len(local))]@np.linalg.inv(M)[:3].T
  again=np.c_[recover_world,np.ones(len(local))]@M[:3].T@K.T;err=float(np.max(np.abs(again[:,:2]/again[:,2:]-uv)));max_error=max(max_error,err)
 assert max_error<=1e-3
 motion_tests.append({'case':'analytic_camera_overtakes_walking_person','camera_relative_z_start_end':relative_z[[0,-1]].tolist(),'constant_body_facing':[0,0,1],'body_velocity':[0,0,.4],'reprojection_roundtrip_max_px':max_error})
 (out/'tests.json').write_text(json.dumps({'scope':'Synthetic asymmetric triangle mesh validates stored camera conventions/view IDs; synthetic paths test measured SE3, not real body accuracy or semantic action recognition','target_cameras':records,'synthetic_moving_camera_tests':motion_tests,'unchanged_runtime':True},indent=2)+'\n')
 print(json.dumps({'asymmetric_camera_cases':2,'camera_views':48,'moving_camera_cases':len(motion_tests),'passed':True}))
if __name__=='__main__':main()
