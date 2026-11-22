"""Train-only native RGB/alpha diagnostics. Never transfer colors across observations."""
import json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from quality_common import NEW,ROOT,BASE,render,camera,rng_state,set_rng,save,sha,load_asset,verify,plan

@torch.no_grad()
def score(model,deform,pipe,data,label,output,exposures=None):
    begin=time.monotonic();candidate=json.loads((NEW/'data'/label/'candidates.json').read_text());assert candidate['train_views']==data['train_views']
    saved=rng_state();flags=deform.reg_spatial_able,deform.reg_temporal_able;deform.reg_spatial_able=deform.reg_temporal_able=False
    rows=[];cores={};zero=torch.zeros(3,device='cuda');n=len(model.get_xyz)
    try:
        for view in data['train_views']:cores[view]=np.load(NEW/'data'/label/f'cores{view:02d}.npz')['labels']
        for record in candidate['scoring']:
            i,view,frame,k=record['candidate_index'],record['view'],record['frame'],record['bin'];assert view not in [0,8,16]
            rgba=np.asarray(Image.open(BASE/'samples'/label/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'),np.float32)/255
            ref=rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:]);refalpha=rgba[...,3];core=cores[view][k]
            cam=camera(data,view,frame);warp=deform.step(model.get_xyz.detach(),cam.fid.unsqueeze(0).expand(n,-1))
            rgbpack=render(cam,model,pipe,zero,warp['d_xyz'],warp['d_rotation'],warp['d_scaling'])
            alpha=render(cam,model,pipe,zero,warp['d_xyz'],warp['d_rotation'],warp['d_scaling'],override_color=torch.ones_like(model.get_xyz))['render'][0].cpu().numpy()
            rgb=rgbpack['render'].permute(1,2,0).cpu().numpy();pred=rgb+.5*(1-alpha[...,None]);error=np.abs(pred-ref).mean(-1)
            world=rgbpack['world_means'];clip=torch.cat([world,torch.ones_like(world[:,:1])],1)@cam.full_proj_transform
            uv=((clip[:,:2]/clip[:,3:]+1)*torch.tensor([352,640],device='cuda')/2-.5).cpu().numpy()
            depth=(torch.cat([world,torch.ones_like(world[:,:1])],1)@cam.world_view_transform)[:,2].cpu().numpy()
            visible=(rgbpack['radii']>0).cpu().numpy()&(depth>0)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<352)&(uv[:,1]>=0)&(uv[:,1]<640)
            groups=np.zeros(n,np.uint8);ids=np.where(visible)[0];xy=np.rint(uv[ids]).astype(int);groups[ids]=core[np.clip(xy[:,1],0,639),np.clip(xy[:,0],0,351)]
            opacity=model.get_opacity.flatten().cpu().numpy();parts=[]
            for part in range(1,7):
                mask=core==part;area=int(mask.sum());assert area==candidate['observations'][i]['areas'][part-1]
                parts.append(dict(part=part-1,area=area,e=float(error[mask].mean()) if area>=64 else None,u=float(np.maximum(refalpha[mask]-alpha[mask],0).sum()/max(float(refalpha[mask].sum()),1e-6)) if area>=64 else None,projected_visible_centers=int((groups==part).sum()),projected_active_centers=int(((groups==part)&(opacity>.005)).sum()),status='valid_proxy' if area>=64 else 'unknown_or_insufficient_support'))
            mask=refalpha>.5;area=int(mask.sum());rows.append(dict(candidate_index=i,view=view,frame=frame,bin=k,exposure=0 if exposures is None else int(exposures[i]),parts=parts,foreground_area=area,foreground_e=float(error[mask].mean()) if area else None,foreground_u=float(np.maximum(refalpha[mask]-alpha[mask],0).sum()/max(float(refalpha[mask].sum()),1e-6)) if area else None))
        medians=[];demand=[]
        for b in range(6):
            values=[r['parts'][b]['e'] for r in rows if r['parts'][b]['e'] is not None];median=float(np.median(values)) if values else None;medians.append(median);cells=[]
            for k in range(8):
                parts=[r['parts'][b] for r in rows if r['bin']==k and r['parts'][b]['e'] is not None]
                cells.append(.5*float(np.median([x['u'] for x in parts]))+.5*float(np.median([x['e']/(x['e']+median+1e-6) for x in parts])) if parts else None)
            demand.append(cells)
        values=[r['foreground_e'] for r in rows if r['foreground_e'] is not None];fullmedian=float(np.median(values)) if values else None
        full={f"{r['view']}:{r['bin']}":None if r['foreground_e'] is None else .5*r['foreground_u']+.5*r['foreground_e']/(r['foreground_e']+fullmedian+1e-6) for r in rows}
        torch.cuda.synchronize();result=dict(domain='training_only',label=label,views=data['train_views'],rows=rows,demand=demand,part_error_medians=medians,full_foreground_error_median=fullmedian,full_difficulty=full,points=n,opacity_gt_0005=int((opacity>.005).sum()),seconds=time.monotonic()-begin,render_calls=2*len(rows),backward_calls=0,optimizer_updates=0,projection_support='Visible raster footprint and projected centers,not surface ownership or occlusion-correct semantic support',candidate_manifest_sha256=sha(NEW/'data'/label/'candidates.json'))
        save(output,result);return result
    finally:deform.reg_spatial_able,deform.reg_temporal_able=flags;set_rng(saved)

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--anchor',choices=['fused','A'],default='fused');args=parser.parse_args();device=verify();p=plan();entry=p['anchors'][args.label][args.anchor];out=NEW/'demand_tables'/f'{args.label}_{args.anchor}_initial.json';assert not out.exists()
    model,deform,pipe,config=load_asset(Path(entry['run']),entry['iteration']);data=json.loads((Path(entry['run'])/'asset_camera_data.json').read_text())
    result=score(model,deform,pipe,data,args.label,out);result['device']=device;result['anchor_snapshot_sha256']=entry['snapshot_sha256'];save(out,result);print(args.label,args.anchor,'initial demand ready',flush=True)
if __name__=='__main__':main()
