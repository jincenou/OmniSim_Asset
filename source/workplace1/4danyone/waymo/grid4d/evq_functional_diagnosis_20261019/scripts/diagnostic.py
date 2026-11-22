"""Read-only posthoc path switches. No optimizer is created or invoked."""
from pathlib import Path
import sys
D=Path(__file__).resolve().parents[1];OLD=D.parent/'early_visual_query_20261019'
sys.path.insert(0,str(OLD/'scripts'))
from assets import *
from evq import pe,projected_samples
import argparse,csv,cv2,lpips,shutil
from evaluate_grid import frozen_functions
from prepare_diagnostics import remap,PARTS

def csvwrite(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=keys);w.writeheader();w.writerows(rows)

def cpu(x):return x.detach().cpu().numpy()

def association(g,q):
    xyz=g.get_xyz;att=torch.cat([pe(q.normalized(xyz)),g._features_dc[:,0],g._scaling,g._rotation,g._opacity],-1);keys=q.anchor_key(q.content);ids=[];ws=[]
    for st in range(0,len(xyz),8192):
        x=q.normalized(xyz[st:st+8192]);v=q.gaussian_key(att[st:st+8192])@keys.T/math.sqrt(64)-(x[:,None]-q.normalized(q.anchors)[None]).square().sum(-1)/(2*.15**2);val,ix=v.topk(4,dim=-1);ids.append(ix);ws.append(val.softmax(-1))
    return torch.cat(ids),torch.cat(ws)

@torch.no_grad()
def trace(q,d,t,pyramids,projection):
    warp,feat=base_features(d,q.anchors,t,False);ctx=q.context(torch.cat([q.content,pe(q.normalized(q.anchors)),pe(t.expand(128,1)),feat],-1));world=native_world(q.anchors,warp['d_xyz']);hom=torch.cat([world,torch.ones_like(world[:,:1])],-1);clip=torch.einsum('mc,vcd->vmd',hom,projection);depth=clip[...,3];safe=torch.where(depth.abs()>1e-6,depth,torch.full_like(depth,1e-6));ndc=clip[...,:2]/safe[...,None]
    off=4*torch.tanh(q.offset(ctx)).reshape(128,4,2,4,2).permute(1,0,2,3,4) if q.arm=='R3' else torch.zeros(4,128,2,4,2,device='cuda');tokens=[];priors=[];valids=[];grids=[]
    for level,inp in enumerate(pyramids):
        h,w=inp.shape[-2:];grid=ndc[:,:,None]+(q.cross[None,None]+off[:,:,level])*inp.new_tensor([2/w,2/h]);valid=(depth>1e-6)[:,:,None]&((grid>=-1)&(grid<=1)).all(-1);val,proj=projected_samples(inp,grid,q.image_projection);tokens.append(proj);priors.append(.05+.95*val[...,3].clamp(0,1));valids.append(valid);grids.append(grid)
    tok=torch.stack(tokens,2).permute(1,0,2,3,4).reshape(128,32,64);valid=torch.stack(valids,2).permute(1,0,2,3).reshape(128,32);prior=torch.stack(priors,2).permute(1,0,2,3).reshape(128,32);null=.05+1-((prior-.05)/.95*valid).mean(-1,keepdim=True);tok=torch.cat([tok,q.null_token[None,None].expand(128,1,64)],1);prior=torch.cat([prior,null],1);valid=torch.cat([valid,torch.ones(128,1,dtype=torch.bool,device='cuda')],1)
    vals=q.value(tok).reshape(128,33,2,32).transpose(1,2)
    if q.arm=='R3':logit=torch.einsum('mhd,mhkd->mhk',q.query(ctx).reshape(128,2,32),q.key(tok).reshape(128,33,2,32).transpose(1,2))/math.sqrt(32)+prior.log()[:,None]
    else:logit=prior.log()[:,None].expand(-1,2,-1)
    weight=logit.masked_fill(~valid[:,None],-torch.inf).softmax(-1);out=q.z_norm(ctx+q.output((weight[...,None]*vals).sum(2).reshape(128,64)))
    return out,dict(grids=cpu(torch.stack(grids,2)),attention=cpu(weight),valid=cpu(valid),anchor_world=cpu(world),anchor_ndc=cpu(ndc),offsets_feature_pixels=cpu(off))

def project_world(x,cam):
    clip=torch.cat([x,torch.ones_like(x[:,:1])],1)@cam.full_proj_transform;uv=(clip[:,:2]/clip[:,3:]+1)*x.new_tensor([352,640])/2-.5
    return uv,clip[:,3]

def covariance(g,w):
    # CUDA does NOT renormalize its final wxyz quaternion.
    r,x,y,z=(g.get_rotation+w['d_rotation']).unbind(1);R=torch.stack([1-2*(y*y+z*z),2*(x*y-r*z),2*(x*z+r*y),2*(x*y+r*z),1-2*(x*x+z*z),2*(y*z-r*x),2*(x*z-r*y),2*(y*z+r*x),1-2*(x*x+y*y)],1).reshape(-1,3,3)
    s=g.get_scaling+w['d_scaling'];return (R*s[:,None,:])@(R*s[:,None,:]).transpose(1,2)

def projected_axes(cov,x,cam):
    eye=cam.world_view_transform;h=torch.cat([x,torch.ones_like(x[:,:1])],1)@eye;tx=math.tan(cam.FoVx/2);ty=math.tan(cam.FoVy/2);z=h[:,2];cx=(h[:,0]/z).clamp(-1.3*tx,1.3*tx)*z;cy=(h[:,1]/z).clamp(-1.3*ty,1.3*ty)*z;J=x.new_zeros(len(x),2,3);J[:,0,0]=352/(2*tx)/z;J[:,1,1]=640/(2*ty)/z;J[:,0,2]=-352/(2*tx)*cx/z.square();J[:,1,2]=-640/(2*ty)*cy/z.square();A=J@eye[:3,:3].T;C=A@cov@A.transpose(1,2)+torch.eye(2,device='cuda')[None]*.3
    ev=torch.linalg.eigvalsh(C);return torch.sqrt(ev.clamp_min(0))

def cov_probe(g,pipe,cam,st,cov):
    from render_evq import GaussianRasterizationSettings,GaussianRasterizer
    settings=GaussianRasterizationSettings(image_height=640,image_width=352,tanfovx=math.tan(cam.FoVx/2),tanfovy=math.tan(cam.FoVy/2),bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=cam.world_view_transform,projmatrix=cam.full_proj_transform,sh_degree=g.active_sh_degree,campos=cam.camera_center,prefiltered=False,debug=False)
    packed=torch.stack([cov[:,0,0],cov[:,0,1],cov[:,0,2],cov[:,1,1],cov[:,1,2],cov[:,2,2]],1)
    a=GaussianRasterizer(settings)(means3D=st['means'],means2D=torch.zeros_like(g.get_xyz),shs=None,colors_precomp=torch.ones_like(g.get_xyz),opacities=g.get_opacity,scales=None,rotations=None,cov3D_precomp=packed)[0]
    b=raster(g,pipe,cam,st,0,True)['render'];err=float((a-b).abs().max());assert err<=1e-5,('covariance convention',err);return err

def contribution(g,pipe,cam,st,masks):
    with torch.enable_grad():
        colors=torch.full_like(g.get_xyz,.25,requires_grad=True);w=st['warp'];im=render_final(cam,g,pipe,torch.zeros(3,device='cuda'),w['d_xyz'],w['d_rotation'],w['d_scaling'],override_color=colors,world_means_override=st['means'],features_override=st['sh'])['render'][0];out=[]
        for k,m in enumerate(masks):
            value=(im*torch.as_tensor(m,device='cuda')).sum();out.append(torch.autograd.grad(value,colors,retain_graph=k<len(masks)-1)[0][:,0].detach().clamp_min(0))
    return torch.stack(out)

def weighted(values,weights):
    x=np.asarray(values);w=np.asarray(weights);ok=np.isfinite(x)&np.isfinite(w)&(w>0)
    if not ok.any():return dict(mean=None,p50=None,p90=None,p99=None,valid_mass=0.)
    x=x[ok];w=w[ok];o=np.argsort(x);cum=np.cumsum(w[o]);return dict(mean=float(np.sum(x*w)/w.sum()),p50=float(np.interp(.5*cum[-1],cum,x[o])),p90=float(np.interp(.9*cum[-1],cum,x[o])),p99=float(np.interp(.99*cum[-1],cum,x[o])),valid_mass=float(w.sum()))

def reference(label,v,f):
    rgba=np.asarray(Image.open(BASE/'samples'/label/'data/images'/f'cam{v:02d}'/f'{f:06d}.png'),np.float32)/255
    return rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:]),rgba[...,3]

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();key=a.key;label=key[:3];arm=key[-2:];device=verify();cv2.setNumThreads(1);torch.set_num_threads(2);start=time.monotonic();plan=json.loads((D/'PLAN_DIAGNOSTIC.json').read_text());sel=plan['selections'][label]
    out=D/'results'/key;out.mkdir(exist_ok=True);cache=D/'cache'/key;cache.mkdir(exist_ok=True);g,d,q,Z,cfg,data,meta=load_portable(OLD/'assets'/key);pipe=SimpleNamespace(**cfg['pipeline']);assert not pipe.compute_cov3D_python and not pipe.convert_SHs_python
    for k in PARAMETERS:getattr(g,k).requires_grad_(False)
    for m in [d.grid4d,d.deform,q]:
        m.eval()
        for p in m.parameters():p.requires_grad_(False)
    ids,binding=association(g,q);np.savez_compressed(cache/'static.npz',base_sh=cpu(g.get_features),opacity=cpu(g.get_opacity),canonical=cpu(g.get_xyz),binding_ids=cpu(ids),binding_weights=cpu(binding),L=float(q.length),C0=C0,diagnostic_only=True)
    inp=Inputs(label,data,False) if arm!='R1' else None
    obs={tuple(x) for x in sel['observations']};deps={tuple(x) for x in sel['render_with_previous_dependencies']};frames=sorted(set(f for v,f in deps));frames=frames[:1] if a.smoke else frames
    scalar,crop,(ssim,ssimsha)=frozen_functions();perceptual=lpips.LPIPS(net='alex').cuda().eval();globalrows=[];partrows=[];expandedrows=[];supportrows=[];checks=[];traces=[]
    oldraw={};rois={};core={};flows={};boxes={}
    for v in [0,8,16]:
        with np.load(OLD/'cache/raw'/f'{key}_32999'/f'view{v:02d}.npz') as z:oldraw[v]=(z['rgb'],z['alpha'])
        with np.load(OLD/'data/expanded_roi'/label/f'view{v:02d}.npz') as z:rois[v]={k:z[k] for k in z.files}
        with np.load(ROOT/'audit/roi'/label/f'view{v:02d}.npz') as z:core[v]=z['labels']
        with np.load(ROOT/'audit/roi'/label/f'flow{v:02d}.npz') as z:flows[v]=(z['backward'],z['valid'])
        boxes[v]=crop([reference(label,v,f)[1]>.5 for f in sel['global_times']])
    previous={};seen=0;covcheck=None
    for f in frames:
        assert shutil.disk_usage(D).free>5*2**30,'5GiB write reserve'
        t=g.get_xyz.new_tensor([[f/data['time_denominator']]]);warp,feat=base_features(d,g.get_xyz.detach(),t,False);base=native_world(g.get_xyz,warp['d_xyz']);dp,dc,diag=q.decode(g,t,Z[f],feat,1.);assert torch.equal(ids,diag['association'])
        full=dict(warp=warp,means=base+dp,sh=torch.cat([g.get_features[:,:1]+dc[:,None],g.get_features[:,1:]],1));cov=covariance(g,warp)
        if covcheck is None:covcheck=cov_probe(g,pipe,camera(data,0,f),full,cov)
        eig=torch.linalg.eigvalsh(cov.double());L=float(q.length);good=(eig[:,0]>1e-12*L*L)&(eig[:,-1]/eig[:,0]>0)&(eig[:,-1]/eig[:,0]<1e10);mah=torch.full((len(base),),float('nan'),device='cuda',dtype=torch.float64);cc=cov.double()+torch.eye(3,device='cuda')*1e-8*L*L;mah[good]=(dp[good].double()*torch.linalg.solve(cc[good],dp[good].double()[...,None])[...,0]).sum(1).clamp_min(0).sqrt()
        statepath=cache/f'state_{f:03d}.npz';payload=dict(base_world=cpu(base),delta_world=cpu(dp),delta_rgb=cpu(dc*C0),Z=cpu(Z[f]),covariance=cpu(cov),rotation=cpu(g.get_rotation+warp['d_rotation']),scale=cpu(g.get_scaling+warp['d_scaling']),diagnostic_only=True)
        if inp is not None and f in sel['support_times']:
            ztrace,tr=trace(q,d,t,inp.support(f),inp.projections);online=q.encode(d,t,inp.support(f),inp.projections)[0];err=float((ztrace-online).abs().max());bakederr=float((ztrace-Z[f]).abs().max());assert max(err,bakederr)<=1e-5,(err,bakederr);payload.update(tr);traces.append(dict(frame=f,trace_encoder_max=err,baked_max=bakederr))
            # Support alpha/depth from final original state, solely for ambiguity; no new image guidance.
            for sv in [1,13,7,19]:
                sc=camera(data,sv,f);sp=raster(g,pipe,sc,full,0.,True);payload[f'support_alpha_{sv}']=cpu(sp['render'][0]);payload[f'support_depthsum_{sv}']=cpu(sp['depth'].squeeze());payload[f'support_uv_base_{sv}']=cpu(project_world(base,sc)[0]);payload[f'support_uv_final_{sv}']=cpu(project_world(base+dp,sc)[0]);payload[f'support_z_final_{sv}']=cpu(project_world(base+dp,sc)[1]);payload[f'support_rgb_{sv}']=cpu(raster(g,pipe,sc,full,.5)['render']).transpose(1,2,0)
        np.savez_compressed(statepath,**payload)
        for v in [0,8,16]:
            if (v,f) not in deps:continue
            cam=camera(data,v,f);target,ra=reference(label,v,f);masks=rois[v]['masks'][f];known=rois[v]['known'][f];images=[];alphas=[];radii=None
            for mode,(pos,col) in plan['modes'].items():
                st=dict(warp=warp,means=base+dp if pos else base,sh=full['sh'] if col else g.get_features);rr=raster(g,pipe,cam,st,0.);rgb=cpu(rr['render']).transpose(1,2,0);alpha=cpu(raster(g,pipe,cam,st,0.,True)['render'][0]);pred=np.clip(rgb+.5*(1-alpha[...,None]),0,1);images.append(pred);alphas.append(alpha)
                if mode=='M11':
                    err=max(float(np.max(abs(rgb-oldraw[v][0][f]))),float(np.max(abs(alpha-oldraw[v][1][f]))));assert err<=1e-5,(key,v,f,err);radii=cpu(rr['radii'])
                if (v,f) not in obs:continue
                common=dict(key=key,label=label,arm=arm,mode=mode,view=v,frame=f,in_global=f in sel['global_times'],in_window=v==sel['window']['view'] and sel['window']['start']<=f<=sel['window']['end'])
                if f in sel['global_times']:
                    x0,y0,x1,y1=boxes[v];pa=torch.as_tensor(pred.transpose(2,0,1).copy(),device='cuda')[None];pb=torch.as_tensor(target.transpose(2,0,1).copy(),device='cuda')[None];globalrows.append(dict(**common,**scalar(pred,target,alpha,ra>.5),ssim=float(ssim(pa,pb)),crop_lpips=float(perceptual(pa[:,:,y0:y1,x0:x1]*2-1,pb[:,:,y0:y1,x0:x1]*2-1))))
                error=np.square(pred-target).mean(-1);te=None
                if f>0:
                    assert (v,mode) in previous and previous[v,mode][0]==f-1
                    prev=previous[v,mode][1];pt=reference(label,v,f-1)[0];flow,valid=flows[v];te=np.square((pred-remap(prev,flow[f-1]))-(target-remap(pt,flow[f-1]))).mean(-1)
                for side in [0,1]:
                    m=masks[side];area=int(m.sum());mass=float(ra[m].sum());row=dict(**common,roi=f'expanded_{side}',known=bool(known[side]),pixels=area,reference_alpha_mass=mass,RGB_MSE=None,soft_undercoverage=None,positive_excess=None,local_lpips=None,temporal_mse=None,temporal_valid_pixels=0,temporal_valid_fraction=None)
                    if known[side]:
                        row.update(RGB_MSE=float(error[m].mean()),soft_undercoverage=float(np.maximum(ra-alpha,0)[m].sum()/mass),positive_excess=float(np.maximum(alpha-ra,0)[m].sum()/mass))
                        if int((m&(ra>.5)).sum())>=128:
                            x0,y0,x1,y1=rois[v]['boxes'][side];pa=torch.as_tensor(pred[y0:y1,x0:x1].transpose(2,0,1).copy(),device='cuda')[None];pb=torch.as_tensor(target[y0:y1,x0:x1].transpose(2,0,1).copy(),device='cuda')[None];row['local_lpips']=float(perceptual(pa*2-1,pb*2-1))
                        if te is not None:
                            tm=m&flows[v][1][f-1]&(remap(rois[v]['masks'][f-1,side].astype('uint8'),flows[v][0][f-1],True)>0);tm &= bool(rois[v]['known'][f-1,side]);n=int(tm.sum());row.update(temporal_valid_pixels=n,temporal_valid_fraction=n/area,temporal_mse=float(te[tm].mean()) if n else None)
                    expandedrows.append(row)
                for pi,pn in enumerate(PARTS):
                    m=core[v][f]==pi+1;n=int(m.sum());row=dict(**common,roi=pn,known=n>0,pixels=n,coverage=float(alpha[m].mean()) if n else None,RGB_MSE=float(error[m].mean()) if n else None,temporal_mse=None,temporal_valid_pixels=0,temporal_valid_fraction=None)
                    if te is not None and n:
                        tm=m&flows[v][1][f-1]&(remap(core[v][f-1],flows[v][0][f-1],True)==pi+1);count=int(tm.sum());row.update(temporal_valid_pixels=count,temporal_valid_fraction=count/n,temporal_mse=float(te[tm].mean()) if count else None)
                    partrows.append(row)
            for mi,mode in enumerate(plan['modes']):previous[v,mode]=(f,images[mi])
            ae0=float(np.max(abs(alphas[0]-alphas[2])));ae1=float(np.max(abs(alphas[1]-alphas[3])));assert max(ae0,ae1)<=1e-5
            checks.append(dict(view=v,frame=f,M11_max=err,alpha00_01=ae0,alpha10_11=ae1));np.savez_compressed(cache/f'view{v:02d}_{f:03d}.npz',pred=np.stack(images),alpha=np.stack(alphas),diagnostic_only=True)
            if (v,f) in obs:
                regionlist=[masks[0],masks[1]]+[core[v][f]==p for p in range(1,7)];weights=contribution(g,pipe,cam,full,regionlist);mass=cpu(weights.sum(1));alpha_mass=np.array([alphas[3][m].sum() for m in regionlist]);rel=float(np.max(abs(mass-alpha_mass)/np.maximum(alpha_mass,1)));assert rel<1e-4,('contribution',rel)
                uv0,z0=project_world(base,cam);uv1,z1=project_world(base+dp,cam);axes=cpu(projected_axes(cov,base+dp,cam));np.savez_compressed(cache/f'weights{v:02d}_{f:03d}.npz',weights=cpu(weights),region_names=np.array(['expanded_0','expanded_1']+PARTS),axes_sigma=axes,actual_radii=radii,uv_base=cpu(uv0),uv_final=cpu(uv1),diagnostic_only=True)
                for ri,name in enumerate(['expanded_0','expanded_1']+PARTS):
                    weight=cpu(weights[ri]);row=dict(key=key,view=v,frame=f,roi=name,known=bool(known[ri]) if ri<2 else bool(regionlist[ri].any()),contribution_mass=float(weight.sum()),contribution_relative_error=rel,ill_conditioned_mass=float(weight[~cpu(good)].sum()))
                    vals={'delta_norm_L':cpu(dp.norm(dim=1)/L),'delta_x_L':cpu(dp[:,0]/L),'delta_y_L':cpu(dp[:,1]/L),'delta_z_L':cpu(dp[:,2]/L),'projected_shift_px':cpu((uv1-uv0).norm(dim=1)),'mahalanobis':cpu(mah),'ellipse_minor_sigma_px':axes[:,0],'ellipse_major_sigma_px':axes[:,1],'ellipse_3sigma_area_px':9*math.pi*axes[:,0]*axes[:,1],'saturated_any_component':cpu((dp.abs()/(.03*L)>.95).any(1).float())}
                    for name2,value in vals.items():
                        for k,x in weighted(value,weight).items():row[f'{name2}_{k}']=x
                    supportrows.append(row)
            seen+=1
        print(key,'frame',f,'observations',seen,flush=True)
    csvwrite(out/'global.csv',globalrows);csvwrite(out/'expanded.csv',expandedrows);csvwrite(out/'parts.csv',partrows);csvwrite(out/'displacement.csv',supportrows)
    save(out/('smoke.json' if a.smoke else 'complete.json'),dict(state='complete',key=key,device=device,seconds=time.monotonic()-start,optimizer_calls=0,model_parameters_frozen=True,association_constant=True,forward_once_per_time=True,covariance_probe_max=covcheck,checks=checks,trace_checks=traces,ssim_sha=ssimsha,source_H_never_query_input=True,diagnostic_not_asset=True));print(key,'COMPLETE',flush=True)
if __name__=='__main__':main()
