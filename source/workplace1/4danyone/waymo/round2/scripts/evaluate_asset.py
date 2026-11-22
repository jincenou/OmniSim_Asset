"""Fixed-camera generated-view diagnostics, complete PLY readback, no model filtering."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image,ImageDraw
import torch

ROUND=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROUND/'backend/src'),str(ROUND/'backend')]
from datasets.FreeTime_dataset import FreeTimeParser
from freetime_ops import positions_at_time,temporal_opacity,validate_vanilla_checkpoint
from gsplat import rasterization
from plyfile import PlyData


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def psnr(mse):return -10*math.log10(max(float(mse),1e-12))


def crop_bounds(masks):
    foreground=np.logical_or.reduce(masks);ys,xs=np.where(foreground)
    if not len(xs):raise ValueError('Empty target foreground throughout sampled times')
    h,w=foreground.shape;pad=max(4,round(.05*max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)))
    x0=max(0,int(xs.min())-pad);x1=min(w,int(xs.max())+pad+1)
    y0=max(0,int(ys.min())-pad);y1=min(h,int(ys.max())+pad+1)
    if min(x1-x0,y1-y0)<64:raise ValueError('Foreground crop too small for native-resolution AlexNet LPIPS')
    return x0,y0,x1,y1


def scalar_metrics(predicted,target,alpha,mask):
    err=np.square(predicted-target);positive=int(mask.sum());outside=~mask
    if not positive or not outside.any():raise ValueError('Foreground/background mask is degenerate')
    silhouette=alpha>.5;union=np.logical_or(mask,silhouette).sum()
    return {'full_mse':float(err.mean()),'full_psnr':psnr(err.mean()),
        'foreground_mse':float(err[mask].mean()),'foreground_psnr':psnr(err[mask].mean()),
        'silhouette_iou':float(np.logical_and(mask,silhouette).sum()/max(int(union),1)),
        'outside_alpha_mean':float(alpha[outside].mean()),
        'outside_alpha_per_foreground_pixel':float(alpha[outside].sum()/positive),
        'outside_alpha_mass_fraction':float(alpha[outside].sum()/max(float(alpha.sum()),1e-12)),
        'foreground_pixels':positive}


def ply_readback(path,splats):
    ply=PlyData.read(path);v=ply['vertex'].data;n=len(v)
    def columns(names):return np.stack([v[name] for name in names],axis=-1).copy()
    k=splats['shN'].shape[1]
    arrays={'means':columns(['x','y','z']),'quats':columns([f'rot_{i}' for i in range(4)]),
        'scales':columns([f'scale_{i}' for i in range(3)]),'opacities':v['opacity'].copy(),
        'sh0':columns([f'f_dc_{i}' for i in range(3)])[:,None],
        'shN':columns([f'f_rest_{i}' for i in range(3*k)]).reshape(n,3,k).transpose(0,2,1).copy(),
        'times':v['time'].copy()[:,None],'durations':v['log_duration'].copy()[:,None],
        'velocities':columns([f'velocity_{i}' for i in range(3)])}
    errors={key:float(np.max(np.abs(value-splats[key].detach().cpu().numpy()))) for key,value in arrays.items()}
    if any(value>1e-6 for value in errors.values()):raise ValueError(f'PLY parameter mismatch: {errors}')
    return {key:torch.as_tensor(value,device='cuda') for key,value in arrays.items()},errors,ply.comments


class Renderer:
    def __init__(self,checkpoint,data):
        self.state=torch.load(checkpoint,map_location='cuda',weights_only=False);validate_vanilla_checkpoint(self.state)
        if self.state['n_frames']!=121:raise ValueError('This evaluation protocol is for the 121-frame branch')
        self.s=self.state['splats'];self.parser=FreeTimeParser(str(data),normalize=True,start_frame=0,end_frame=121)
        self.indices={int(cid)-1:index for index,cid in enumerate(self.parser.camera_ids)}
        self.cfg=self.state['training_state']['config']
        if self.state['training_state']['data_factor']!=1:raise ValueError('Specify actual factored master before evaluating')

    @torch.inference_mode()
    def render(self,view,frame,splats=None):
        s=self.s if splats is None else splats;index=self.indices[view];cid=self.parser.camera_ids[index]
        width,height=self.parser.imsize_dict[cid];t=frame/120
        positions=positions_at_time(s['means'],s['velocities'],s['times'],t,self.state['use_velocity'])
        opacity=(s['opacities'].sigmoid()*temporal_opacity(s['times'],s['durations'],t)).clamp_min(1e-4)
        rgb,alpha,_=rasterization(means=positions,quats=s['quats'],scales=s['scales'].exp(),opacities=opacity,
            colors=torch.cat([s['sh0'],s['shN']],1),
            viewmats=torch.linalg.inv(torch.as_tensor(self.parser.camtoworlds[index:index+1],device='cuda',dtype=torch.float32)),
            Ks=torch.as_tensor(self.parser.Ks_dict[cid][None],device='cuda',dtype=torch.float32),
            width=width,height=height,sh_degree=self.cfg['sh_degree'],packed=self.cfg['packed'],
            near_plane=self.cfg['near_plane'],far_plane=self.cfg['far_plane'],
            rasterize_mode='antialiased' if self.cfg['antialiased'] else 'classic')
        return rgb[0].cpu().numpy(),alpha[0,...,0].cpu().numpy()


def cuda_alpha_probe():
    from foreground_loss import composite_foreground
    observations=[]
    for background in (0.,.5,1.):
        opacity_logit=torch.tensor([0.],device='cuda',requires_grad=True)
        rgb,alpha,_=rasterization(means=torch.tensor([[.6,.6,2.]],device='cuda'),
            quats=torch.tensor([[1.,0.,0.,0.]],device='cuda'),scales=torch.full((1,3),.15,device='cuda'),
            opacities=opacity_logit.sigmoid(),colors=torch.zeros((1,3),device='cuda'),
            viewmats=torch.eye(4,device='cuda')[None],Ks=torch.tensor([[[16.,0.,16.],[0.,16.,16.],[0.,0.,1.]]],device='cuda'),
            width=32,height=32,packed=True)
        target_rgb=torch.ones_like(rgb);target_alpha=torch.zeros_like(alpha);target_alpha[:,4:12,4:12]=1
        pred,target,alpha_loss=composite_foreground(rgb,alpha,target_rgb,target_alpha,torch.full((1,3),background,device='cuda'))
        loss=(pred-target).abs().mean()+.1*alpha_loss;loss.backward()
        gradient=float(opacity_logit.grad.item())
        if not math.isfinite(gradient) or gradient<=0:raise ValueError('Outside black Gaussian lacks removal gradient')
        observations.append({'background':background,'opacity_logit_gradient':gradient})
    return {'state':'actual_cuda_rasterizer_gradient_passed','cases':observations,
        'loss':'full-image L1 plus 0.1 alpha L1; isolated synthetic black Gaussian outside foreground; no target-mask clipping',
        'scope':'Checks rasterizer/compositing gradient path, not the full training loss including SSIM/LPIPS'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();physical=int(os.environ['CUDA_VISIBLE_DEVICES'])
    if physical not in (4,5,6,7):raise ValueError('Unauthorized GPU')
    expected=subprocess.check_output(['nvidia-smi',f'--id={physical}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-')
    torch.cuda.init();actual=str(torch.cuda.get_device_properties(0).uuid)
    if actual!=expected or torch.cuda.device_count()!=1:raise RuntimeError('Actual CUDA UUID mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'device.json',{'physical_gpu':physical,'cuda_uuid':actual,'pid':os.getpid()})
    save(args.output/'cuda_alpha_probe.json',cuda_alpha_probe())
    renderer=Renderer(args.checkpoint,args.data)
    ply=args.checkpoint.parent.parent/f'{args.checkpoint.stem}.ftgs.ply'
    restored,errors,comments=ply_readback(ply,renderer.s)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval()
    frames=list(range(0,121,10));views=[0,8,16,1,9,17];rows=[];temporal=[];inputs=[];readback_render_errors=[];crops={}
    for view in views:
        paths=[args.data/'images'/f'cam{view:02d}'/f'{frame:06d}.png' for frame in frames]
        rgba=[np.asarray(Image.open(path).convert('RGBA'),dtype=np.float32)/255 for path in paths]
        masks=[image[...,3]>.5 for image in rgba];crop=crop_bounds(masks);crops[str(view)]=crop
        x0,y0,x1,y1=crop;previous=None
        for frame,path,image,mask in zip(frames,paths,rgba,masks,strict=True):
            raw,alpha=renderer.render(view,frame)
            if raw.shape!=image[...,:3].shape:raise ValueError('Native supervision and render sizes differ')
            if frame in (0,60,120):
                pr,pa=renderer.render(view,frame,restored)
                difference=max(float(np.abs(pr-raw).max()),float(np.abs(pa-alpha).max()))
                if difference>1e-5:raise ValueError('PLY independent-array render differs from checkpoint')
                readback_render_errors.append(difference)
            panels=[];gray_pair=None
            for bg in (0.,.5,1.):
                predicted=np.clip(raw+(1-alpha[...,None])*bg,0,1)
                target=image[...,:3]*image[...,3:]+(1-image[...,3:])*bg
                metrics=scalar_metrics(predicted,target,alpha,mask)
                tensors=[torch.from_numpy(np.ascontiguousarray(v[y0:y1,x0:x1])).permute(2,0,1)[None].cuda() for v in (predicted,target)]
                with torch.inference_mode():value=float(lpips(*tensors).item())
                lpips.reset()
                metrics.update(view=view,frame=frame,seconds=frame/25,background=bg,
                    split='validation_generated' if view in (0,8,16) else 'training_diagnostic',crop_lpips=value,crop_xyxy=crop)
                rows.append(metrics);panels.extend([target,predicted])
                if bg==.5:gray_pair=(predicted,target)
            if previous is not None:
                pp,pt,pm=previous;pred,target=gray_pair;region=mask|pm
                pd=pred-pp;td=target-pt
                temporal.append({'view':view,'frame':frame,'interval_seconds':.4,
                    'prediction_mean_abs_change':float(np.abs(pd).mean()),'target_mean_abs_change':float(np.abs(td).mean()),
                    'foreground_union_delta_mae':float(np.abs(pd-td)[region].mean()),
                    'meaning':'fixed-camera sampled temporal difference, not optical flow or proof of natural motion'})
            previous=(*gray_pair,mask)
            inputs.append({'view':view,'frame':frame,'path':str(path),'sha256':sha(path)})
            if frame in (0,60,120):
                sheet=Image.fromarray(np.round(np.concatenate(panels,axis=1)*255).astype(np.uint8))
                ImageDraw.Draw(sheet).text((4,4),f'v{view} frame{frame}: target/render pairs on black, gray, white',fill='red')
                sheet.save(args.output/f'view{view:02d}_frame{frame:03d}.jpg',quality=94)
    aggregates={}
    keys=['full_psnr','foreground_psnr','crop_lpips','silhouette_iou','outside_alpha_mean','outside_alpha_per_foreground_pixel','outside_alpha_mass_fraction']
    for split in ('validation_generated','training_diagnostic'):
        aggregates[split]={str(bg):{key:float(np.mean([row[key] for row in rows if row['split']==split and row['background']==bg])) for key in keys} for bg in (0.,.5,1.)}
    result={'state':'evaluated_generated_observations_visual_review_pending','checkpoint':str(args.checkpoint),'checkpoint_sha256':sha(args.checkpoint),
        'steps':renderer.state['step']+1,'gaussians':len(renderer.s['means']),'frames':frames,'views':views,'crops':crops,
        'aggregates':aggregates,'samples':rows,'temporal':temporal,'inputs':inputs,
        'ply_roundtrip':{'all_parameter_max_abs':errors,'render_max_abs':max(readback_render_errors),'sampled_renders':len(readback_render_errors),'comments':comments},
        'protocol':{'mask':'saved predicted SAM2/BiRefNet RGBA alpha; threshold >0.5',
            'composition':'straight target RGB multiplied once by alpha; predicted raster RGB already premultiplied',
            'crop':'per-view union of foreground over all13 sampled times plus5% margin; identical across runs with same master; no resizing',
            'psnr':'mean per-image PSNR; MSE floor1e-12 (120dB); foreground uses target mask, full-image and outside alpha also reported',
            'no_point_filtering':True,'reference':'generated observations, not real Waymo new-view ground truth'}}
    save(args.output/'metrics.json',result);print(json.dumps({'state':result['state'],'aggregates':aggregates}),flush=True)


if __name__=='__main__':main()
