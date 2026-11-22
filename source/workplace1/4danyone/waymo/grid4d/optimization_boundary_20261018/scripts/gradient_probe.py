"""Independent loss-path gradients and conditional local projected support; no steps."""
import argparse
import json
import sys
import time
import numpy as np
import torch
from PIL import Image
from boundary_common import *
from utils.loss_utils import ssim
PARTS=['head','torso','left_arm','right_arm','left_leg','right_leg']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--run');parser.add_argument('--iteration',type=int,default=48000);args=parser.parse_args();begin=time.monotonic();device=verify();sys.addaudithook(protect_inputs)
    spec=plan();source=Path(args.run) if args.run else Path(spec['identities'][args.label]['start_run'])
    destination=NEW/'research/gradients'/args.label/('start' if not args.run else source.parent.name+'_'+source.name+f'_{args.iteration}')
    assert not destination.with_suffix('.json').exists()
    model,deform,pipe,config=load_asset(source,args.iteration);data=json.loads((source/'asset_camera_data.json').read_text());named=groups(model,deform);params=sum(named.values(),[])
    train=json.loads(Path(spec['identities'][args.label]['data']['training']).read_text());paths={(r['view'],r['frame']):r['path'] for r in train}
    anchors=[a for a in json.loads((NEW/'data/anchors.json').read_text()) if a['label']==args.label];rows=[];renders=backwards=0
    for anchor in anchors:
        v,f=anchor['view'],anchor['frame']
        with Image.open(paths[v,f]) as image:rgba=torch.as_tensor(np.asarray(image).copy(),device='cuda',dtype=torch.float32).permute(2,0,1)/255
        target=rgba[:3]*rgba[3:]+.5*(1-rgba[3:]);cam=camera(data,v,f,target);refalpha=rgba[3]
        partition=torch.tensor(np.load(NEW/'data'/args.label/'bands'/f'view{v:02d}.npy',mmap_mode='r')[f].copy(),device='cuda')
        core=np.load(ROOT/'optimization/data'/args.label/f'view{v:02d}.npy',mmap_mode='r')[f];region=core==PARTS.index(anchor['part'])+1
        warp=deform.step(model.get_xyz.detach(),cam.fid.unsqueeze(0).expand(len(model.get_xyz),-1));package=render(cam,model,pipe,torch.full((3,),.5,device='cuda'),warp['d_xyz'],warp['d_rotation'],warp['d_scaling'])
        alpha=render(cam,model,pipe,torch.zeros(3,device='cuda'),warp['d_xyz'],warp['d_rotation'],warp['d_scaling'],override_color=torch.ones_like(model.get_xyz))['render'][0];renders+=2
        image=package['render'];losses=dict(rgb=.8*(image-target).abs().mean()+.2*(1-ssim(image,target)),old_alpha=spec['lambda_alpha']*old_alpha(alpha,refalpha),uniform_extra=spec['delta']*extra_alpha(alpha,refalpha,partition,'U')[0],band_extra=spec['delta']*extra_alpha(alpha,refalpha,partition,'B')[0])
        clip=torch.cat([package['world_means'].detach(),torch.ones_like(model.get_xyz[:,:1])],1)@cam.full_proj_transform
        uv=((clip[:,:2]/clip[:,3:]+1)*torch.tensor([352.,640.],device='cuda')/2-.5).detach().cpu().numpy();visible=package['visibility_filter'].detach().cpu().numpy()
        inside=visible&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<352)&(uv[:,1]>=0)&(uv[:,1]<640)
        ids=np.where(inside)[0];local=np.zeros(len(uv),bool);x=np.clip(np.rint(uv[ids,0]).astype(int),0,351);y=np.clip(np.rint(uv[ids,1]).astype(int),0,639);local[ids]=region[y,x]
        norms={}
        for name,loss in losses.items():
            grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);backwards+=1;offset=0;pergroup={}
            for group,parameters in named.items():
                values=grads[offset:offset+len(parameters)];offset+=len(parameters);connected=[g for g in values if g is not None];assert all(bool(torch.isfinite(g).all()) for g in connected)
                pergroup[group]=dict(l2=float(torch.stack([g.square().sum() for g in connected]).sum().sqrt()) if connected else 0.,connected_tensors=len(connected),total_tensors=len(parameters))
            norms[name]=dict(weighted_loss=float(loss),groups=pergroup)
        area=int(region.sum());rows.append(dict(anchor=anchor,reference_core_pixels=area,unknown_foreground_pixels=int(((core==0)&(rgba[3].detach().cpu().numpy()>=.5)).sum()),loss_gradients=norms,frustum_nonzero_radius_points=int(visible.sum()),low_opacity_points=int((model.get_opacity<.01).sum()),local_center_projected_points=int(local.sum()),local_low_opacity_points=int((local&(model.get_opacity.detach().cpu().numpy().ravel()<.01)).sum()),coverage=float(alpha[torch.tensor(region,device='cuda')].mean()) if area else None))
        del losses,grads,warp,package,alpha,image
    assert all(p.grad is None for p in params)
    save(destination.with_suffix('.json'),dict(state='passed',label=args.label,source=str(source),iteration=args.iteration,device=device,rows=rows,seconds=time.monotonic()-begin,render_calls=renders,autograd_calls=backwards,optimizer_steps=0,normalization='Weighted component L2 norm over all participating parameters; new alpha terms have equal pixel weight mass, not equal gradient norm',support_scope='Nonzero radius and reference-core center projection are conditional support, not proven unoccluded material correspondence'))
    print(args.label,source.name,args.iteration,'gradient paths recorded',flush=True)


if __name__=='__main__':main()
