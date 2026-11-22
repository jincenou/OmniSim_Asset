"""Frozen generated-heldout metrics; only complete final comparisons are published."""
import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
import lpips
from grid_common import ROOT,BASE,load_asset,raw_render,camera,manifest,sha,save,verify,stamp


def frozen_functions():
    path=BASE/'scripts/evaluate_branch.py';tree=ast.parse(path.read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['scalar','crop','ssim_function']]
    scope=dict(np=np,math=math,ast=ast,torch=torch,R=BASE,sha=sha)
    exec(compile(tree,str(path),'exec'),scope)
    return scope['scalar'],scope['crop'],scope['ssim_function']()


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);parser.add_argument('--iteration',required=True,type=int)
    args=parser.parse_args();device=verify();run=ROOT/'runs'/f'{args.label}_33k_v1'
    iteration=args.iteration;assert iteration in [5000,15000,33000]
    snapshot=json.loads((run/'snapshots'/f'{iteration}.json').read_text())
    for file,digest in snapshot['files'].items():assert sha(run/file)==digest
    gs,deform,pipe,config=load_asset(run,iteration)
    data,_=manifest(args.label)
    # Read model-independent camera metadata from the exported asset package itself.
    camera_data=json.loads((run/'asset_camera_data.json').read_text())
    refs=np.load(run/'snapshots'/f'{iteration}_reference.npz');replay=[]
    for key in refs.files:
        view,frame=key.split('_');rgb,alpha=raw_render(gs,deform,pipe,camera(camera_data,int(view[1:]),float(frame[1:])))
        delta=float(np.abs(torch.cat([rgb,alpha]).cpu().numpy()-refs[key]).max());assert delta<1e-5
        replay.append(delta)
    scalar,crop,(ssim,ssim_sha)=frozen_functions()
    perceptual=lpips.LPIPS(net='alex').cuda().eval()
    sampled=np.rint(np.linspace(0,data['frame_count']-1,13)).astype(int).tolist()
    stage=ROOT/'tmp/evaluations'/f'{args.label}_{iteration}';stage.mkdir(parents=True,exist_ok=False)
    is_final=iteration==33000
    spec=importlib.util.spec_from_file_location('frozen_video',BASE/'backend/src/video_io.py');video=importlib.util.module_from_spec(spec);spec.loader.exec_module(video)
    rows=[];dense=[];videos={};crops={}
    if is_final:
        cache=ROOT/'tmp/grid_raw'/args.label;cache.mkdir(parents=True,exist_ok=False)
    for view in [0,8,16]:
        def target_path(frame):return BASE/'samples'/args.label/'data/images'/f'cam{view:02d}'/f'{frame:06d}.png'
        masks=[np.asarray(Image.open(target_path(frame)))[...,3]>127 for frame in sampled]
        box=crop(masks);crops[view]=box;x0,y0,x1,y1=box
        old={}
        if is_final:
            for branch in ['ftgs_opt15k','4dgs_grid25','4dgs_grid61']:
                p=ROOT/'tmp/baseline_raw'/args.label/branch
                assert (p/'complete.json').exists()
                file=np.load(p/f'view{view:02d}.npz')
                old[branch]=(file['rgb'],file['alpha']);file.close()
            writer=video.MP4Writer(str(stage/f'view{view:02d}.mp4'),25,352*5,672)
        raw_frames=[];alpha_frames=[]
        try:
            for frame in range(data['frame_count']) if is_final else sampled:
                rgba=np.asarray(Image.open(target_path(frame)),np.float32)/255
                target=rgba[...,:3]*rgba[...,3:]+.5*(1-rgba[...,3:]);mask=rgba[...,3]>.5
                raw,alpha=raw_render(gs,deform,pipe,camera(camera_data,view,frame))
                raw=raw.permute(1,2,0).cpu().numpy();alpha=alpha[0].cpu().numpy()
                assert np.isfinite(raw).all() and np.isfinite(alpha).all()
                pred=np.clip(raw+.5*(1-alpha[...,None]),0,1)
                values=dict(view=view,frame=frame,seconds=frame/25,**scalar(pred,target,alpha,mask))
                values['foreground_soft_coverage']=float(alpha[mask].mean())
                dense.append(values.copy())
                if frame in sampled:
                    a=torch.tensor(pred.transpose(2,0,1).copy(),device='cuda')[None]
                    b=torch.tensor(target.transpose(2,0,1).copy(),device='cuda')[None]
                    values.update(ssim=float(ssim(a,b)),crop_lpips=float(perceptual(a[:,:,y0:y1,x0:x1]*2-1,b[:,:,y0:y1,x0:x1]*2-1)),
                        crop_xyxy=box,input_sha256=sha(target_path(frame)))
                    rows.append(values)
                if is_final:
                    raw_frames.append(raw);alpha_frames.append(alpha)
                    images=[target]+[np.clip(r[frame]+.5*(1-a[frame,...,None]),0,1) for r,a in old.values()]+[pred]
                    strip=np.rint(np.concatenate(images,1)*255).astype(np.uint8)
                    canvas=Image.new('RGB',(352*5,672),'#282828');canvas.paste(Image.fromarray(strip),(0,32));draw=ImageDraw.Draw(canvas)
                    names=['Generated reference','FTGS opt15k','HUST grid25 33k','HUST grid61 33k','Grid4D 33k']
                    for i,name in enumerate(names):draw.text((352*i+5,9),f'{name} | {args.label} v{view:02d} f{frame}',fill='white')
                    writer.append_data(np.asarray(canvas))
                    if frame in [0,(data['frame_count']-1)//2,data['frame_count']-1]:canvas.save(stage/f'view{view:02d}_frame{frame:03d}.jpg',quality=95)
        finally:
            if is_final:writer.close()
        if is_final:
            np.savez_compressed(cache/f'view{view:02d}.npz',rgb=np.stack(raw_frames),alpha=np.stack(alpha_frames))
            p=stage/f'view{view:02d}.mp4';cap=cv2.VideoCapture(str(p));count=0
            while True:
                ok,image=cap.read()
                if not ok:break
                assert image.shape==(672,1760,3);count+=1
            cap.release();assert count==data['frame_count']
            videos[p.name]=dict(frames=count,sha256=sha(p))
    keys=['full_psnr','foreground_psnr','ssim','crop_lpips','silhouette_iou','outside_alpha_mean']
    assert len(rows)==39
    report=dict(state='complete_generated_heldout_quality_requires_visual_review',label=args.label,iteration=iteration,
        actual_optimizer_updates=iteration-1,device=device,at=stamp(),mean={k:float(np.mean([v[k] for v in rows])) for k in keys},
        samples=rows,dense_samples=dense,independent_replay_max_error=max(replay),videos=videos,
        protocol=dict(name='random8-v1.1-generated-heldout',background=.5,resolution=[352,640],heldout_views=[0,8,16],
            sampled_frames=sampled,crops=crops,ssim_source_sha256=ssim_sha,lpips='lpips0.1.4 AlexNet v0.1; [-1,1]',
            reference='Frozen generated reference, development evaluation, not real Waymo novel-view ground truth',
            same_gpu_compute_budget=False,training_background_difference='Grid4D gray0.5; historical HUST white1.0'),
        asset=dict(run=str(run),camera_data=str(run/'asset_camera_data.json'),configuration=str(run/'config.json'),
            canonical=str(run/'point_cloud'/f'iteration_{iteration}'/'point_cloud.ply'),hash_and_deform=str(run/'deform'/f'iteration_{iteration}'/'deform.pth'),
            replay_entry=str(ROOT/'scripts/validate_grid.py'),source_commit=config['upstream_commit'],patch=str(ROOT/'research/source.patch')))
    save(stage/'metrics.json',report)
    final=ROOT/'results'/args.label/f'{iteration}';final.parent.mkdir(parents=True,exist_ok=True);assert not final.exists()
    stage.rename(final)
    if is_final:save(cache/'complete.json',dict(state='complete',source_results=str(final/'metrics.json'),views=[0,8,16],frames=data['frame_count'],temporary=True))
    print(json.dumps(dict(label=args.label,iteration=iteration,mean=report['mean'])),flush=True)


if __name__=='__main__':main()
