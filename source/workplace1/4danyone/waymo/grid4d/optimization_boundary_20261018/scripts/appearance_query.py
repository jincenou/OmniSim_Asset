"""Train-only SH conflict probes and conditional canonical-coordinate donor audit."""
import argparse
import csv
import json
import sys
import time
import cv2
import numpy as np
import torch
from PIL import Image,ImageDraw
from boundary_common import *
from utils.general_utils import quaternion_to_matrix
PARTS=['head','torso','left_arm','right_arm','left_leg','right_leg']


def write_csv(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def crop_box(mask,pad=8):
    ys,xs=np.where(mask)
    if not len(xs):return None
    return [max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),min(352,int(xs.max())+pad+1),min(640,int(ys.max())+pad+1)]


def panel(path,images,names,box):
    if box is None:box=[0,0,352,640]
    x0,y0,x1,y1=box;w,h=x1-x0,y1-y0
    canvas=Image.new('RGB',(w*len(images),h+36),'#242424');draw=ImageDraw.Draw(canvas)
    for i,(image,name) in enumerate(zip(images,names)):
        pixels=np.rint(np.clip(image,0,1)*255).astype(np.uint8)
        if pixels.ndim==2:pixels=np.repeat(pixels[...,None],3,-1)
        canvas.paste(Image.fromarray(pixels[y0:y1,x0:x1]),(i*w,36));draw.text((i*w+3,3),name,fill='white')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def lookup(image,uv,nearest=False):
    return cv2.remap(image.astype(np.float32),uv[:,0].astype(np.float32).reshape(-1,1),uv[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0).reshape((len(uv),)+image.shape[2:])


class Probe:
    def __init__(self,label):
        self.label=label;spec=plan();run=Path(spec['identities'][label]['start_run']);self.model,self.deform,self.pipe,self.config=load_asset(run,48000)
        self.data=json.loads((run/'asset_camera_data.json').read_text());rows=json.loads(Path(spec['identities'][label]['data']['training']).read_text());self.paths={(r['view'],r['frame']):r['path'] for r in rows};self.refs={};self.rendered={};self.cores={}
        for params in groups(self.model,self.deform).values():
            for p in params:p.requires_grad_(False)
        self.model._features_dc.requires_grad_(True);self.model._features_rest.requires_grad_(True);self.render_calls=0;self.backward_calls=0

    def reference(self,view,frame):
        assert view not in [0,8,16]
        key=view,frame
        if key not in self.refs:
            with Image.open(self.paths[key]) as image:self.refs[key]=np.asarray(image,np.float32)/255
        return self.refs[key]

    def core(self,view,frame,part):
        if view not in self.cores:self.cores[view]=np.load(ROOT/'optimization/data'/self.label/f'view{view:02d}.npy',mmap_mode='r')
        return self.cores[view][frame]==PARTS.index(part)+1

    @torch.no_grad()
    def rendered_view(self,view,frame):
        key=view,frame
        if key in self.rendered:return self.rendered[key]
        cam=camera(self.data,view,frame);n=len(self.model.get_xyz);warp=self.deform.step(self.model.get_xyz,cam.fid.unsqueeze(0).expand(n,-1));black=torch.zeros(3,device='cuda')
        args=(cam,self.model,self.pipe,black,warp['d_xyz'],warp['d_rotation'],warp['d_scaling'])
        rgb=render(*args);a=render(*args,override_color=torch.ones_like(self.model.get_xyz))['render'][0];bound=self.deform.grid4d.bound
        canonical=render(*args,override_color=(self.model.get_xyz/bound+1)/2)['render'];canonical=((canonical/a.clamp_min(1e-8)[None])*2-1)*bound
        self.render_calls+=3
        value=dict(rgb=rgb['render'].permute(1,2,0).cpu().numpy(),alpha=a.cpu().numpy(),canonical=canonical.permute(1,2,0).cpu().numpy(),depth=(rgb['depth'].squeeze()/a.clamp_min(1e-8)).cpu().numpy(),camera=cam)
        self.rendered[key]=value;return value

    @torch.no_grad()
    def project(self,canonical,view,frame):
        cam=camera(self.data,view,frame);xyz=torch.tensor(canonical,device='cuda',dtype=torch.float32);warp=self.deform.step(xyz,cam.fid.unsqueeze(0).expand(len(xyz),-1));d=warp['d_xyz']
        world=(quaternion_to_matrix(d[:,:4])@xyz[...,None]).squeeze(-1)+d[:,4:];homogeneous=torch.cat([world,torch.ones_like(world[:,:1])],1)
        clip=homogeneous@cam.full_proj_transform;uv=(clip[:,:2]/clip[:,3:]+1)*torch.tensor([352.,640.],device='cuda')/2-.5
        depth=(homogeneous@cam.world_view_transform)[:,2]
        return uv.cpu().numpy(),depth.cpu().numpy()

    def gradient(self,view,frame,part):
        rgba=self.reference(view,frame);mask=self.core(view,frame,part);area=int(mask.sum())
        if not area:return dict(area=0,mse=None,norm=0.,grad=None,support=None)
        cam=camera(self.data,view,frame);count=len(self.model.get_xyz)
        with torch.no_grad():warp=self.deform.step(self.model.get_xyz,cam.fid.unsqueeze(0).expand(count,-1))
        package=render(cam,self.model,self.pipe,torch.full((3,),.5,device='cuda'),warp['d_xyz'],warp['d_rotation'],warp['d_scaling']);self.render_calls+=1
        target=torch.tensor((rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:])).transpose(2,0,1).copy(),device='cuda');selector=torch.tensor(mask,device='cuda')
        loss=(package['render'][:,selector]-target[:,selector]).square().mean()
        grads=torch.autograd.grad(loss,[self.model._features_dc,self.model._features_rest]);self.backward_calls+=1;gradient=torch.cat(grads,1).detach().cpu();magnitude=gradient.square().sum((1,2)).sqrt();support=(magnitude>1e-12)&package['visibility_filter'].detach().cpu()
        return dict(area=area,mse=float(loss),norm=float(gradient.square().sum().sqrt()),grad=gradient,support=support)

    def mapping(self,target_view,target_frame,donor,mask,part):
        # No target reference RGB is accepted or used by this function.
        target=self.rendered_view(target_view,target_frame);view,frame=donor['view'],donor['frame'];other=self.rendered_view(view,frame)
        ys,xs=np.where(mask);positions=np.stack([xs,ys],1).astype(np.float32);canonical=target['canonical'][mask]
        selfuv,_=self.project(canonical,target_view,target_frame);uv,z=self.project(canonical,view,frame)
        donor_a=lookup(other['alpha'],uv);donor_z=lookup(other['depth'],uv);donor_canonical=lookup(other['canonical'],uv)
        back,_=self.project(donor_canonical,target_view,target_frame);cycle=np.linalg.norm(back-positions,axis=1);selfcycle=np.linalg.norm(selfuv-positions,axis=1)
        zerror=np.abs(z-donor_z);tolerance=.01+.03*np.abs(z)
        inbounds=(uv[:,0]>=0)&(uv[:,0]<=351)&(uv[:,1]>=0)&(uv[:,1]<=639)&np.isfinite(uv).all(1)&(z>0)
        donor_mask=lookup(self.core(view,frame,part).astype(np.float32),uv,True)>.5
        donor_refalpha=lookup(self.reference(view,frame)[...,3],uv)
        stages=dict(target_alpha=target['alpha'][mask]>=.5,self_projection=selfcycle<=2.,donor_inbounds=inbounds,donor_alpha=donor_a>=.5,donor_reference_alpha=donor_refalpha>=.5,same_part_core=donor_mask,depth_consistent=zerror<=tolerance,cycle_consistent=cycle<=2.)
        accepted=np.logical_and.reduce(list(stages.values()));reliability=np.exp(-.5*(cycle/2)**2)*np.minimum(target['alpha'][mask],donor_a)/(1+zerror/np.maximum(tolerance,1e-8));reliability[~accepted]=0
        assert np.isfinite(reliability).all()
        # Only after geometry-based eligibility/ranking is computed, read donor colors.
        rgba=self.reference(view,frame);donor_rgb=rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:]);colors=lookup(donor_rgb,uv)
        return dict(colors=colors,accepted=accepted,reliability=reliability,cycle=cycle,self_cycle=selfcycle,depth_error=zerror,stages={k:int(v.sum()) for k,v in stages.items()},positions=positions)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args();begin=time.monotonic();device=verify();sys.addaudithook(protect_inputs);cv2.setNumThreads(2)
    out=NEW/'results/analysis'/args.label;out.mkdir(parents=True,exist_ok=False);protocol=json.loads((NEW/'research/analysis_protocol.json').read_text());probe=Probe(args.label)
    anchors=[a for a in json.loads((NEW/'data/anchors.json').read_text()) if a['label']==args.label];appearances=[];candidates=[];results=[];audit=[]
    for anchor in anchors:
        view,frame,part=anchor['view'],anchor['frame'],anchor['part'];mask=probe.core(view,frame,part);area=int(mask.sum());box=crop_box(mask)
        target=probe.rendered_view(view,frame);reconstructed=np.clip(target['rgb']+.5*(1-target['alpha'][...,None]),0,1);base_grad=probe.gradient(view,frame,part)
        audit.append(dict(**anchor,reference_core_pixels=area,core_sha256=__import__('hashlib').sha256(mask.tobytes()).hexdigest(),crop_xyxy=box,reference_path=probe.paths[view,frame],reference_sha256=sha(probe.paths[view,frame]),gradient_mse=base_grad['mse'],status='measured' if area else 'no_reliable_core'))
        maps=[]
        for rank,donor in enumerate(anchor['donors']):
            assert (donor['view']==view) != (donor['frame']==frame)
            other=probe.gradient(donor['view'],donor['frame'],part);common=None;cosine=inner=None;status='no_reference_core'
            if base_grad['grad'] is not None and other['grad'] is not None:
                common=base_grad['support']&other['support'];n=int(common.sum());status='insufficient_common_contribution'
                if n>=protocol['minimum_common_contributing_gaussians']:
                    a=base_grad['grad'][common].double().flatten();b=other['grad'][common].double().flatten();norm_a=float(a.norm());norm_b=float(b.norm())
                    if min(norm_a,norm_b)>protocol['gradient_zero_norm']:
                        inner=float(a@b);cosine=inner/(norm_a*norm_b);status='measured'
                    else:status='zero_gradient'
            appearances.append(dict(anchor=anchor['id'],label=args.label,part=part,target_view=view,target_frame=frame,donor_view=donor['view'],donor_frame=donor['frame'],axis=donor['type'],target_mse=base_grad['mse'],donor_mse=other['mse'],target_gradient_norm=base_grad['norm'],donor_gradient_norm=other['norm'],common_contributing_gaussians=int(common.sum()) if common is not None else 0,cosine=cosine,inner_product=inner,negative_inner_product=inner<0 if inner is not None else None,status=status))
            mapping=probe.mapping(view,frame,donor,mask,part) if area else None
            if mapping is not None:maps.append(mapping)
            donor_rgba=probe.reference(donor['view'],donor['frame']);donor_core=probe.core(donor['view'],donor['frame'],part);donor_rgb=donor_rgba[...,:3]*donor_rgba[...,3:]+.5*(1-donor_rgba[...,3:]);gray=cv2.cvtColor(donor_rgb,cv2.COLOR_RGB2GRAY);lap=cv2.Laplacian(gray,cv2.CV_32F)
            pixels=int(mapping['accepted'].sum()) if mapping else 0
            candidate=dict(anchor=anchor['id'],rank=rank,part=part,target_view=view,target_frame=frame,donor=donor,roi_pixels=area,transfer_pixels=pixels,transfer_fraction=pixels/area if area else None,stage_pass_counts=mapping['stages'] if mapping else {},donor_core_pixels=int(donor_core.sum()),donor_laplacian_energy=float(np.square(lap[donor_core]).mean()) if donor_core.any() else None,cycle_mean_qualified=float(mapping['cycle'][mapping['accepted']].mean()) if pixels else None,depth_error_mean_qualified=float(mapping['depth_error'][mapping['accepted']].mean()) if pixels else None,qualification='conditional_geometry_pass' if pixels>=64 else 'insufficient_qualified_pixels',failure_reason=None if pixels>=64 else 'reference_core_missing' if not area else 'fewer_than64_pixels_after_alpha_part_depth_and_bidirectional_projection_gates')
            candidates.append(candidate)
            mapped=reconstructed.copy();valid_image=np.zeros(mask.shape,np.float32)
            if mapping is not None:
                local=mapped[mask];local[mapping['accepted']]=mapping['colors'][mapping['accepted']];mapped[mask]=local;valid_image[mask]=mapping['accepted']
            # This diagnostic panel shows eligibility, never a modified Gaussian asset.
            panel(out/'donors'/f"{anchor['id']}_donor{rank:02d}.png",[reconstructed,mapped,valid_image],['Reconstruction','Conditional donor','Qualified pixels'],box)
        rgba=probe.reference(view,frame);reference=rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:]);variants={'current':reconstructed.copy()};coverage={}
        if area:
            base=reconstructed[mask];nearest=base.copy();uniform=base.copy();weighted=base.copy();used=np.zeros(area,bool);sums=np.zeros_like(base);weights=np.zeros(area);weighted_sum=np.zeros_like(base);number=np.zeros(area,int)
            for mapping in maps:
                valid=mapping['accepted']
                if int(valid.sum())<64:continue
                selected=valid&~used;nearest[selected]=mapping['colors'][selected];used|=valid
                sums[valid]+=mapping['colors'][valid];number+=valid;weighted_sum+=mapping['colors']*mapping['reliability'][:,None];weights+=mapping['reliability']
            ok=number>0;uniform[ok]=sums[ok]/number[ok,None];ok_weight=weights>0;weighted[ok_weight]=weighted_sum[ok_weight]/weights[ok_weight,None]
            for name,value,valid in [('nearest',nearest,used),('uniform',uniform,ok),('reliability',weighted,ok_weight)]:
                image=reconstructed.copy();image[mask]=value;variants[name]=image;coverage[name]=float(valid.mean())
        else:
            variants.update({name:reconstructed.copy() for name in ['nearest','uniform','reliability']});coverage={name:None for name in ['nearest','uniform','reliability']}
        metrics={name:dict(mse=float(np.square(image[mask]-reference[mask]).mean()) if area else None,mae=float(np.abs(image[mask]-reference[mask]).mean()) if area else None) for name,image in variants.items()}
        target_gray=cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY);target_lap=cv2.Laplacian(target_gray,cv2.CV_32F)
        for candidate in candidates:
            if candidate['anchor']==anchor['id']:candidate['donor_has_higher_laplacian_energy_than_target']=candidate['donor_laplacian_energy']>float(np.square(target_lap[mask]).mean()) if area and candidate['donor_laplacian_energy'] is not None else None
        results.append(dict(anchor=anchor['id'],roi_pixels=area,crop=box,metrics=metrics,transfer_coverage=coverage,target_defect='reconstruction_reference_core_mse_and_visual_panel',target_laplacian_energy=float(np.square(target_lap[mask]).mean()) if area else None,correspondence_scope='Conditional opacity-weighted canonical expectation; cycle/depth/part consistency is not material-ground-truth verification',ranking_target_color_access=False))
        panel(out/'donors'/f"{anchor['id']}_comparison.png",[reference]+list(variants.values()),['Reference','Current','Nearest','Uniform','Reliability'],box)
    assert all(p.grad is None for params in groups(probe.model,probe.deform).values() for p in params)
    write_csv(out/'appearance_time_probe.csv',appearances)
    save(out/'anchors.json',dict(state='frozen_train_only_anchors_audited',anchors=audit,heldout_content_reads=0))
    save(out/'query_readiness.json',dict(state='offline_diagnostic_complete_visual_review_pending',label=args.label,device=device,candidates=candidates,anchors=results,correspondence_verified=dict(level='conditional_model_geometry_only',ground_truth=False,qualified_candidates=sum(c['qualification']=='conditional_geometry_pass' for c in candidates),checked_candidates=len(candidates)),complementary_information=dict(level='evaluate_qualified_transfers_and_visual_structure',donor_sharpness_is_not_proof=True),simple_transfer_gain=[dict(anchor=r['anchor'],nearest_mse_gain=r['metrics']['current']['mse']-r['metrics']['nearest']['mse'] if r['roi_pixels'] else None,uniform_mse_gain=r['metrics']['current']['mse']-r['metrics']['uniform']['mse'] if r['roi_pixels'] else None,reliability_mse_gain=r['metrics']['current']['mse']-r['metrics']['reliability']['mse'] if r['roi_pixels'] else None) for r in results],query_training_executed=False,optimizer_steps=0,render_calls=probe.render_calls,backward_calls=probe.backward_calls,seconds=time.monotonic()-begin,next_minimum_suggestion='Review same-surface/identity and transfer failures before proposing learned Query; all transfers remain offline'))
    print(args.label,'SH probes and actual donor transfer audit complete',flush=True)


if __name__=='__main__':main()
