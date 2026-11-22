import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--replay',action='store_true');a=p.parse_args()
import numpy as np,torch
from runtime import *
OUT=ROOT/'research/preflight';OUT.mkdir(exist_ok=True)
record=device_record()
c={'width':96,'height':160,'K':[[140,0,43.2],[0,145,77.1],[0,0,1]],'c2w':np.eye(4).tolist()}
if a.replay:
 m,d=load(OUT/'checkpoint.pt');images=[]
 with torch.no_grad():
  for t in [0,.173,.5,1]:
   o=render(m,c,t);images.append(torch.cat([o['render'],o['alpha']]).cpu())
 got=torch.stack(images);ref=torch.load(OUT/'reference.pt',weights_only=True);err=float((got-ref).abs().max());assert err<1e-6
 save_json(OUT/'replay.json',{'device':record,'max_abs':err,'times':[0,.173,.5,1],'new_process':True});sys.exit()
# Arithmetic and the matching KNN extension.
x=torch.arange(256,device='cuda',dtype=torch.float32);assert float((x*x).sum())==sum(i*i for i in range(256))
from simple_knn._C import distCUDA2
rng=np.random.default_rng(71);xyz=rng.normal(size=(256,3)).astype('float32')*.12;xyz[:,2]+=2
knn=distCUDA2(torch.tensor(xyz,device='cuda'));assert torch.isfinite(knn).all() and (knn>0).all()
np.savez(OUT/'synthetic.npz',points=xyz,colors=rng.uniform(.1,.9,(256,3)).astype('float32'),time=rng.uniform(0,1,(256,1)).astype('float32'))
m=create(OUT/'synthetic.npz',10)
with torch.no_grad():
 m._scaling[:]=torch.log(torch.tensor([.027,.042,.061],device='cuda'));m._rotation[:,1:]=.07*torch.randn_like(m._rotation[:,1:]);m._rotation_r[:,1:]=.05*torch.randn_like(m._rotation_r[:,1:])
# Actual camera pixel projection; off-center rectangular synthetic and all real cameras.
proj_errors=[]
for cam in [c]+json.loads((ROOT/'objects/OBJ_0018/clips/C001/manifest.json').read_text())['cameras']:
 cc=Camera(cam,.4);q=np.c_[xyz,np.ones(len(xyz))]@np.linalg.inv(cam['c2w']).T;k=np.asarray(cam['K']);uv=(q[:,:3]@k.T);uv=uv[:,:2]/uv[:,2:]
 clip=torch.tensor(np.c_[xyz,np.ones(len(xyz))],device='cuda',dtype=torch.float32)@cc.full_proj_transform;ndc=clip[:,:2]/clip[:,3:];pixels=((ndc+1)*torch.tensor([cam['width'],cam['height']],device='cuda')-1)/2
 err=float(np.max(abs(pixels.cpu().numpy()-uv)));assert err<.005;proj_errors.append(err)
 assert np.allclose(cc.camera_center.cpu(),np.array(cam['c2w'])[:3,3])
fields=['_xyz','_t','_scaling','_scaling_t','_rotation','_rotation_r','_opacity','_features_dc'];before={f:getattr(m,f).detach().clone() for f in fields}
o=render(m,c,.37);assert o['render'].shape==(3,160,96);assert o['alpha'].shape==(1,160,96)
loss=(o['render']-.3).square().mean()+o['alpha'].square().mean();loss.backward();grads={}
for f in fields:
 g=getattr(m,f).grad;assert g is not None and torch.isfinite(g).all();grads[f]=float(g.abs().max());assert grads[f]>0,f
m.optimizer.step();updates={f:float((getattr(m,f)-before[f]).abs().max()) for f in fields};assert min(updates.values())>0
# Alpha supervision separately reaches geometry/opacity.
m.optimizer.zero_grad(set_to_none=True);render(m,c,.37)['alpha'].square().mean().backward();assert float(m._opacity.grad.abs().max())>0
with torch.no_grad():
 images=[torch.cat([o['render'],o['alpha']]).cpu() for t in [0,.173,.5,1] for o in [render(m,c,t)]]
 images=torch.stack(images);delta=float((images[0]-images[-1]).abs().max());assert delta>1e-4
checkpoint(OUT/'checkpoint.pt',m,{'steps':10},1,rng);torch.save(images,OUT/'reference.pt')
subprocess.run([sys.executable,__file__,'--replay'],check=True,cwd=ROOT)
save_json(OUT/'result.json',{'state':'passed','device':record,'knn':True,'projection_max_px':max(proj_errors),'finite_nonzero_gradients':grads,'parameter_updates':updates,'time_query_max_delta':delta,'replay':json.loads((OUT/'replay.json').read_text()),'scope':'Synthetic native4D render/backward + real camera numerical projections; not training quality'})
print(json.dumps(json.loads((OUT/'result.json').read_text())),flush=True)
