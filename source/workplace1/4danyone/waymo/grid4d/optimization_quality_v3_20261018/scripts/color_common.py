"""Fixed-geometry SH study. Inspired by Instant Colorization's contribution blocks.

Official reference bytes and original research license: ../source/instant_color/.
This implements native residual gradients, warm SH, and trace damping per PLAN;
it does not copy the reference's SH reset or static COLMAP execution path.
"""
import os,shutil
from quality_common import *
from utils.sh_utils import eval_sh

def setup():
    spec=plan();source=Path(spec['color']['source']);model,deform,pipe,config=load_asset(source,53000)
    assert not pipe.convert_SHs_python and model.active_sh_degree==3
    data=json.loads((source/'asset_camera_data.json').read_text())
    for name in PARAMETERS:getattr(model,name).requires_grad_(name in ['_features_dc','_features_rest'])
    for p in deform.grid4d.parameters():p.requires_grad_(False)
    for p in deform.deform.parameters():p.requires_grad_(False)
    frozen={n:getattr(model,n).detach().clone() for n in PARAMETERS if n not in ['_features_dc','_features_rest']}
    frozen['hash']={k:v.detach().clone() for k,v in deform.grid4d.state_dict().items()};frozen['deform']={k:v.detach().clone() for k,v in deform.deform.state_dict().items()}
    return source,model,deform,pipe,config,data,frozen

def pool(data):
    full,_=manifest('N11');frames=sorted(set(round(k*(data['frame_count']-1)/7) for k in range(8)));assert len(frames)==8
    observations=[]
    for view in data['train_views']:
        for frame in frames:
            path=Path(full['source_data'])/'images'/f'cam{view:02d}'/f'{frame:06d}.png'
            with Image.open(path) as im:rgba=torch.tensor(np.asarray(im).copy(),device='cuda',dtype=torch.float32).permute(2,0,1)/255
            target=rgba[:3]*rgba[3:]+.5*(1-rgba[3:]);observations.append((camera(data,view,frame,target),str(path)))
    assert len(observations)==168
    return observations

def package(model,deform,pipe,cam,color=None):
    with torch.no_grad():warp=deform.step(model.get_xyz.detach(),cam.fid.unsqueeze(0).expand(len(model.get_xyz),-1))
    return render(cam,model,pipe,torch.full((3,),.5,device='cuda'),warp['d_xyz'],warp['d_rotation'],warp['d_scaling'],override_color=color)

def basis_for(model,cam,world):
    dirs=world.detach()-cam.camera_center;dirs=dirs/dirs.norm(dim=1,keepdim=True);k=(model.active_sh_degree+1)**2
    basis=eval_sh(model.active_sh_degree,torch.eye(k,device='cuda').expand(len(dirs),-1,-1),dirs)
    return basis

def contribution(model,deform,pipe,cam):
    constant=torch.full_like(model.get_xyz,.25,requires_grad=True);pkg=package(model,deform,pipe,cam,constant)
    weight=torch.autograd.grad(pkg['render'][0].sum(),constant)[0][:,0].detach()
    assert bool(torch.isfinite(weight).all()) and float(weight.min())>=0
    return weight,basis_for(model,cam,pkg['world_means'])

def verify_frozen(model,deform,frozen):
    for n in PARAMETERS:
        if n in frozen:equal(getattr(model,n),frozen[n])
    equal(deform.grid4d.state_dict(),frozen['hash']);equal(deform.deform.state_dict(),frozen['deform'])
    assert model.active_sh_degree==3

@torch.no_grad()
def full_loss(model,deform,pipe,observations):
    values=[]
    for cam,path in observations:
        image=package(model,deform,pipe,cam)['render'];values.append(float((image-cam.original_image).square().mean()))
    result=float(np.mean(values));assert np.isfinite(result)
    return result

def save_color(run,epoch,updates,model,deform,pipe,data,optimizer,costs):
    start=time.monotonic();saved=rng_state();ply=run/'point_cloud'/f'iteration_{epoch}'/'point_cloud.ply';assert not ply.exists();model.save_ply(str(ply))
    deform_path=run/'deform'/f'iteration_{epoch}'/'deform.pth'
    if epoch==0:deform.save_weights(str(run),epoch)
    else:
        deform_path.parent.mkdir(parents=True,exist_ok=True);os.link(run/'deform/iteration_0/deform.pth',deform_path)
    refs=asset_reference(model,deform,pipe,data);(run/'snapshots').mkdir(exist_ok=True);np.savez_compressed(run/'snapshots'/f'{epoch}_reference.npz',**refs)
    recovery=run/'recoveries'/f'color_pass_{epoch}.pt';recovery.parent.mkdir(exist_ok=True)
    atomic_torch(recovery,dict(color_pass=epoch,color_updates=updates,base_grid4d_updates=52999,features_dc=model._features_dc.detach(),features_rest=model._features_rest.detach(),color_optimizer=None if optimizer is None else optimizer.state_dict(),rng=saved,costs=dict(costs),frozen_asset_files=dict(ply=str(ply.relative_to(run)),deform=str(deform_path.relative_to(run))),quality_plan_sha256=sha(NEW/'PLAN.json')))
    save(run/'snapshots'/f'{epoch}.json',dict(iteration=epoch,color_pass=epoch,color_optimizer_updates=updates,actual_optimizer_updates=52999,counter_note='Iteration directory denotes color pass, not additional Grid4D updates',active_sh_degree=model.active_sh_degree,points=len(model.get_xyz),files={str(p.relative_to(run)):sha(p) for p in [ply,deform_path,recovery]}))
    equal(rng_state(),saved);return time.monotonic()-start
