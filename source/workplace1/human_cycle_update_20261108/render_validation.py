from pathlib import Path
import json,math,numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as imageio
from hasim.shared.assets.library_v2 import AssetLibrary,load_asset
from hasim.shared.assets.human_cycle import CycleCatalog,CyclicRuntimeAsset
from hasim.shared.insertion import SceneState,AssetInstance
from hasim.simulation.rendering.insertion import GaussianLayerRenderer,WorldCamera
base=Path('/mnt/ojc/workplace1/human_cycle_update_20261108');lib=AssetLibrary('/mnt/ojc/workplace1/asset_library');catalog=CycleCatalog.load(base/'render_catalog.json');renderer=GaussianLayerRenderer('cuda:0');cameras=[]
for name,eye in [('front_oblique',[3,2.5,1.2]),('side',[0,4,1.05])]:
 origin=np.asarray(eye,float);forward=np.array([0.,0.,.85])-origin;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(forward,right);matrix=np.eye(4);matrix[:3,:3]=np.stack([right,down,forward],axis=1);matrix[:3,3]=origin;cameras.append(WorldCamera(name,384,512,np.array([[850,0,192],[0,850,256],[0,0,1.]]),matrix))
def floor(cam):
 v,u=np.indices((cam.height,cam.width));ray=np.stack([(u-192)/850,(v-256)/850,np.ones_like(u)],-1)@cam.world_from_camera[:3,:3].T;origin=cam.world_from_camera[:3,3];depth=-origin[2]/np.where(abs(ray[...,2])>1e-9,ray[...,2],1e-9);pts=origin+ray*depth[...,None];checker=(np.floor(pts[...,0]/.25)+np.floor(pts[...,1]/.25))%2;gray=np.where(checker[...,None]==0,215,235);image=np.repeat(gray,3,axis=2);image[depth<=0]=245;return image
backgrounds={c.name:floor(c) for c in cameras}
for identity in catalog.profiles:
 suffix=identity.split('_')[-1]
 runtime=load_asset(lib.get(identity));profile=catalog.get(runtime.asset.id);cycle=CyclicRuntimeAsset(runtime,profile);out=base/'gpu_cycles'/suffix;out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True);fps=15;times=np.arange(math.ceil(max(4.0,3*profile.period_s)*fps)+1)/fps;writer=imageio.get_writer(out/'cycles.mp4',fps=fps,codec='libx264',quality=8,macro_block_size=1);rows=[]
 try:
  for i,t in enumerate(times):
   state=SceneState((AssetInstance('human',runtime.asset.id,'human',cycle.sample(float(t)),np.eye(4),float(t)),));layers=renderer.render(state,tuple(cameras));panel=Image.new('RGB',(768,540),'#1b2730');draw=ImageDraw.Draw(panel)
   for j,cam in enumerate(cameras):
    layer=layers[cam.name];rgb=np.clip(layer['rgb']+backgrounds[cam.name]*(1-layer['alpha'][...,None]),0,255).astype(np.uint8);panel.paste(Image.fromarray(rgb),(j*384,28));draw.text((j*384+5,7),f'{suffix} {cam.name} t={t:.2f} phase={t%profile.period_s:.3f}',fill='white')
   panel.save(out/'frames'/f'{i:05d}.png');writer.append_data(np.asarray(panel));rows.append(dict(t=float(t),source_samples=cycle.source_samples(float(t)),gaussian_count=len(state.instances[0].cloud.means)))
   if i%20==0:print(suffix,i,len(times),flush=True)
 finally:writer.close()
 (out/'report.json').write_text(json.dumps(dict(profile=profile.to_dict(),fps=fps,frames=rows),indent=2)+'\n');print(suffix,'complete',len(times),flush=True)
