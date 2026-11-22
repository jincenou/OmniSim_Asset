"""Traceable foreground metrics for the complete two-scene pilot cohort."""
from __future__ import annotations
import argparse
import asyncio
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import sys

B=Path(__file__).resolve().parents[1]
N=Path(os.environ.get('ASSET_EVALUATION_ROOT',str(B/'frontend_revision_01')))
R=Path('/mnt/ojc/workplace1/reconviagen_demo')
OUT=N/'evaluation/assets'
REGISTRY=Path(os.environ.get('ASSET_METRIC_REGISTRY',str(N/'evaluation/metric_registry.json')))
RESULT_NAME=os.environ.get('ASSET_METRIC_RESULT_NAME','metrics')
P=Path('/mnt/ojc/DriveLab')
sys.path[:0]=[str(P),str(R/'scripts')]
from asset_extraction.io_utils import read_json,write_json
from asset_extraction.generation import sha256,digest,GpuPool


def extent_diagnostics(extent_lwh, reference_lwh):
    """Length-aligned box proportions, not metric scale recovery or surface IoU."""
    import math
    extent=list(map(float,extent_lwh));reference=list(map(float,reference_lwh))
    if len(extent)!=3 or len(reference)!=3 or any(not math.isfinite(v) or v<=0 for v in extent+reference):
        raise ValueError('Positive finite L/W/H required')
    scale=reference[0]/extent[0]
    aligned=[v*scale for v in extent]
    errors=[100*(a/b-1) for a,b in zip(aligned,reference)]
    intersection=math.prod(min(a,b) for a,b in zip(aligned,reference))
    union=math.prod(aligned)+math.prod(reference)-intersection
    return dict(reference_lwh_m=reference,length_aligned_lwh_m=aligned,uniform_scale=scale,
                width_signed_percent=errors[1],height_signed_percent=errors[2],
                Width_APE_percent=abs(errors[1]),Height_APE_percent=abs(errors[2]),
                Box_extent_IoU=intersection/union)


def registry(*,partial=False):
    report=read_json(N/'pilot_results/report.json')
    if report['status'] not in {'completed','completed_with_failures'} and not partial:
        raise ValueError('Evaluation registry requires both full pilots to finish')
    rows=[]
    for scene in report['scenes']:
        dataset,scene_id=scene['dataset'],scene['scene_id']
        run=N/f'runs/{dataset}_pilot'
        sources={r['asset_id']:r for r in read_json(run/f'scenes/{scene_id}/logs/export_items_full.json')}
        gt=read_json(N/f'evaluation/matches/{scene_id}.json')
        matches={r['obj_id']:r for r in gt['thresholds']['0.5']['query']['matches']}
        for item in report['assets']:
            if item['dataset']!=dataset or not item['published']:
                continue
            source=sources[item['obj_id']]
            anchor=next(r for r in source['real_images'] if r['source_image']==source['source']['source_image']
                and max(abs(a-b) for a,b in zip(r['bbox_xyxy'],source['source']['bbox_xyxy']))<.01)
            match=matches.get(item['obj_id'])
            native=match['native_annotation'] if match else None
            # Semantic orientation is required; no best-view search or fitting to output.
            eligible=bool(native and native.get('pose_link') and item['canonical']['heading_verified']
                          and item['route']=='H')
            rows.append(dict(obj_id=item['obj_id'],dataset=dataset,scene_id=scene_id,route=item['route'],
                category=item['category'],quality_status=item['status'],canonical=item['canonical'],
                ply=item['published']['canonical_path'],ply_sha256=item['published']['sha256'],
                source_rgba=anchor['rgba_path'],source_sha256=sha256(anchor['rgba_path']),
                source_crop_box=anchor['crop_box'],matched_annotation=native if eligible else None,
                matched_exclusion=None if eligible else 'Native source pose and verified vehicle heading are both required'))
    from evaluate_asset_metrics import CLIP_PATH,DINO_PATH
    checkpoints={name:dict(path=str(path),sha256=sha256(path)) for name,path in
        [('CLIP',CLIP_PATH),('DINOv3',DINO_PATH),
         ('FID',R/'cache/fid/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth')]}
    protocol=dict(version='scene_pilot_foreground_v1',script_sha256=sha256(__file__),checkpoints=checkpoints,
        metric_helpers_sha256={p.name:sha256(p) for p in
            [R/'scripts/evaluate_asset_metrics.py',R/'scripts/evaluate_pixel_metrics.py']},
        queries=len(report['assets']),exported_payloads=len(rows),partial_snapshot=partial,
        source='One original SAM foreground per admitted query, not a generated FLUX image; source-conditioned, not heldout.',
        orbit='16 azimuths 0..337.5 degrees, step22.5, elevation12, canonical z-up; shared canonical angles, not source-relative phase.',
        preprocessing='512px white render, alpha>0.5; tight foreground square with 5% per-side padding; white composite; bicubic512.',
        similarity='Mean 16 source/render cosine similarities within each query, then equal-weight query mean.',
        CLIP='OpenAI CLIP ViT-B/32; standard224 transform on protocol512 crop; normalized global image feature.',
        DINO='DINOv3 ViT-H/16+ LVD1689M, local timm weights; float32; I16 normalized CLS.',
        ED_R='1-cosine of DINOv3 foreground-patch means, only native matched camera with verified vehicle heading; independently centered isotropic normalized foregrounds.',
        pixels='Same normalized matched foregrounds as ED-R; PSNR RGB[0,1], Gaussian SSIM sigma1.5, LPIPS-Alex v0.1; source-view diagnostics only.',
        matched_camera='Native cuboid pose and camera calibration; isotropic scale by native cuboid length; no generated-output camera search. Cuboid grounding approximation.',
        FID='pytorch-fid0.3.0 Inception2048 float32/noTF32; one real source vs16 rendered views per valid query; float64 unbiased covariance via exact low-rank SVD.',
        FID_limits='Exploratory per-scene distribution score; correlated views, small/source-biased sample, source vs orbit mismatch. Not insertion quality. No FID if fewer than2 identities.',
        cohorts='All valid published exports including quality reject/review; final keep subset separately. H versus B0 populations differ, so scores are not causal route comparisons.',
        failure_policy='All321 queries retained in generation/AOR reports. Feature errors remain explicit, not silently counted as successful.',
        posthoc_best_view_selection=False,FAD='not calculated, as instructed')
    path=REGISTRY
    document=dict(protocol=protocol,assets=rows)
    if path.exists() and read_json(path)!=document:
        raise ValueError('Frozen metric registry changed')
    write_json(path,document)
    return document


def render(indices):
    import numpy as np
    import torch
    from PIL import Image
    from gsplat import rasterization
    from asset_extraction.export import inspect_ply
    sys.path.insert(0,'/mnt/wmj/workspace4/pipeline')
    from curate.core import load_rigid
    torch.set_num_threads(4)
    rows=read_json(REGISTRY)['assets']
    def tensor(a):return torch.as_tensor(a,device='cuda',dtype=torch.float32)
    for index in indices:
        row=rows[index];out=OUT/row['obj_id'];out.mkdir(parents=True,exist_ok=True)
        state=out/'render.json'
        if state.exists():
            old=read_json(state)
            if old.get('row_signature')!=digest(row):
                raise ValueError('Metric cache identity changed')
            continue
        try:
            if sha256(row['ply'])!=row['ply_sha256'] or sha256(row['source_rgba'])!=row['source_sha256']:
                raise ValueError('Metric input hash mismatch')
            inspect_ply(row['ply'])
            rgba=Image.open(row['source_rgba']).convert('RGBA')
            source=Image.new('RGB',rgba.size,'white');source.paste(rgba,mask=rgba.getchannel('A'))
            source.save(out/'source.png');rgba.getchannel('A').save(out/'source_alpha.png')
            gs=load_rigid(row['ply']);lo,hi=np.quantile(gs.xyz[0],[.005,.995],axis=0)
            center=(lo+hi)/2;radius=np.linalg.norm(hi-lo)/2;distance=radius*2.2/.78
            focal=.5*512*distance/radius*.78;K=np.array([[focal,0,256],[0,focal,256],[0,0,1]])
            quats=tensor(gs.rot[0]);quats/=quats.norm(dim=-1,keepdim=True)
            arrays=[tensor(gs.xyz[0]),quats,tensor(gs.scale[0]),tensor(gs.opacity),tensor(gs.color)]
            def image(view,intrinsic,w,h,name,scale=1.):
                with torch.inference_mode():
                    rgb,alpha,_=rasterization(arrays[0]*scale,arrays[1],arrays[2]*scale,*arrays[3:],
                        tensor(view)[None],tensor(intrinsic)[None],w,h,backgrounds=torch.ones((1,3),device='cuda'),packed=False)
                if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():
                    raise ValueError('Nonfinite render')
                rgb=rgb[0,:,:,:3].clamp(0,1).cpu().numpy();alpha=alpha[0,:,:,0].clamp(0,1).cpu().numpy()
                Image.fromarray((rgb*255).round().astype('uint8')).save(out/f'{name}.png')
                Image.fromarray((alpha*255).round().astype('uint8')).save(out/f'{name}_alpha.png')
                return int((alpha>.5).sum())
            counts=[]
            for view_id in range(16):
                az=view_id*np.pi/8;el=np.deg2rad(12)
                direction=np.array([np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)])
                forward=-direction;right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
                rotation=np.stack([right,np.cross(forward,right),forward]);view=np.eye(4)
                view[:3,:3]=rotation;view[:3,3]=-rotation@(center+distance*direction)
                counts.append(image(view,K,512,512,f'orbit_{view_id:02d}'))
            dimensions=None;dimension_error=None
            reference=row.get('native_dimensions_annotation')
            if reference:
                try:dimensions=extent_diagnostics(hi-lo,reference['size_lwh'])
                except (ValueError,KeyError,TypeError) as exc:dimension_error=str(exc)
            matched=None;annotation=row['matched_annotation']
            if annotation:
                scale=annotation['size_lwh'][0]/(hi[0]-lo[0])
                grounding=np.eye(4);grounding[2,3]=-annotation['size_lwh'][2]/2
                view=np.linalg.inv(annotation['camera_to_world'])@np.asarray(annotation['object_to_world'])@grounding
                x0,y0,x1,y1=row['source_crop_box'];w,h=rgba.size
                crop=np.array([[w/(x1-x0),0,-x0*w/(x1-x0)],[0,h/(y1-y0),-y0*h/(y1-y0)],[0,0,1]])
                intrinsic=crop@np.asarray(annotation['camera_intrinsic'])
                count=image(view,intrinsic,w,h,'matched_source',scale)
                matched=dict(foreground_pixels=count,scale_by_length=scale,view=view.tolist(),K=intrinsic.tolist(),
                    image_size=[w,h],pose_link=annotation['pose_link'])
            write_json(state,dict(status='completed',row_signature=digest(row),ply_sha256=row['ply_sha256'],source_sha256=row['source_sha256'],
                all_16_nonempty=all(v>0 for v in counts),alpha_threshold=.5,foreground_pixels=counts,matched=matched,
                gaussian_center_quantile_lwh=(hi-lo).tolist(),dimensions=dimensions,dimension_error=dimension_error))
            print('render',row['obj_id'],min(counts),flush=True)
            del arrays,gs;torch.cuda.empty_cache()
        except Exception as exc:
            write_json(state,dict(status='failed',error=f'{type(exc).__name__}: {exc}',
                row_signature=digest(row),ply_sha256=row['ply_sha256'],source_sha256=row['source_sha256']))


def features(encoder):
    import numpy as np
    import torch
    from evaluate_asset_metrics import Encoders,foreground
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if encoder=='fid':
        os.environ['TORCH_HOME']=str(R/'cache/fid');sys.path.insert(0,str(R/'tools/fid_deps'))
        from pytorch_fid.inception import InceptionV3
        weight=R/'cache/fid/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth'
        assert sha256(weight)=='6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'
        model=InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).cuda().eval()
    else:model=Encoders(encoder)
    for row in read_json(REGISTRY)['assets']:
        out=OUT/row['obj_id'];audit=read_json(out/'render.json');target=out/f'{encoder}.npz'
        if target.exists():continue
        if audit['status']!='completed' or not audit['all_16_nonempty']:continue
        names=['source']+[f'orbit_{i:02d}' for i in range(16)]
        if audit['matched'] and audit['matched']['foreground_pixels']>0:names+=['matched_source']
        images,masks=[],[]
        for name in names:
            im,mask,_=foreground(out/f'{name}.png',out/f'{name}_alpha.png');images.append(im);masks.append(mask)
        values=defaultdict(list)
        batch_size=2 if encoder=='dino' else 16
        for start in range(0,len(images),batch_size):
            if encoder=='fid':
                batch=torch.from_numpy(np.stack([np.asarray(im,dtype=np.float32)/255 for im in images[start:start+batch_size]])).permute(0,3,1,2).cuda()
                with torch.inference_mode():result={'global_embedding':model(batch)[0].flatten(1).cpu().numpy()}
            else:result=model.encode(images[start:start+batch_size],masks[start:start+batch_size])
            for key,value in result.items():values[key].append(value)
        values={key:np.concatenate(value) for key,value in values.items()}
        if any(not np.isfinite(value).all() for value in values.values()):raise ValueError('Nonfinite features')
        temporary=target.with_suffix('.tmp.npz');np.savez(temporary,names=np.array(names),**values);temporary.replace(target)
        print(encoder,row['obj_id'],len(names),flush=True)


def pixels():
    import numpy as np
    import torch,lpips
    from evaluate_asset_metrics import foreground
    from evaluate_pixel_metrics import compute
    os.environ['TORCH_HOME']=str(R/'cache/pixel_metrics_torch')
    torch.set_num_threads(4)
    net=lpips.LPIPS(net='alex',version='0.1',lpips=True,pnet_rand=False,pnet_tune=False,verbose=False).cuda().eval()
    for row in read_json(REGISTRY)['assets']:
        out=OUT/row['obj_id'];audit=read_json(out/'render.json');target=out/'pixels.json'
        if target.exists():continue
        if audit['status']!='completed' or not audit['matched'] or not audit['matched']['foreground_pixels']:continue
        a,am,_=foreground(out/'source.png',out/'source_alpha.png')
        b,bm,_=foreground(out/'matched_source.png',out/'matched_source_alpha.png')
        write_json(target,compute(a,am,b,bm,net));print('pixels',row['obj_id'],flush=True)


def fid_value(x,y):
    import numpy as np
    if len(x)<2 or len(y)<2:return None
    x=x.astype('float64');y=y.astype('float64');mx=x.mean(0);my=y.mean(0)
    a=(x-mx)/np.sqrt(len(x)-1);b=(y-my)/np.sqrt(len(y)-1)
    if len(x)*len(y)<=4_000_000:
        cross=float(np.linalg.svd(a@b.T,compute_uv=False).sum())
    else:
        cx=a.T@a;cy=b.T@b
        eigen,vectors=np.linalg.eigh((cx+cx.T)/2)
        root=(vectors*np.sqrt(np.maximum(eigen,0)))@vectors.T
        middle=root@cy@root
        cross=float(np.sqrt(np.maximum(np.linalg.eigvalsh((middle+middle.T)/2),0)).sum())
    value=float(((mx-my)**2).sum()+(a*a).sum()+(b*b).sum()-2*cross)
    if value < -1e-6:raise ValueError('Numerically invalid negative FID')
    return max(0.,value)


def summarize():
    import numpy as np
    registry_data=read_json(REGISTRY);rows=[]
    for row in registry_data['assets']:
        out=OUT/row['obj_id'];audit=read_json(out/'render.json')
        record={k:row[k] for k in ['obj_id','dataset','scene_id','category','route','quality_status']}
        record.update(render_status=audit['status'],render_error=audit.get('error'),
                      ply_present_at_summary=Path(row['ply']).is_file(),
                      valid_16_views=audit['status']=='completed' and audit.get('all_16_nonempty',False),
                      CLIP_I16=None,DINO_I16=None,ED_R=None)
        dimensions=audit.get('dimensions') or {}
        record.update({key:dimensions.get(key) for key in ['Width_APE_percent','Height_APE_percent','Box_extent_IoU']})
        record['dimension_evidence']=dimensions or None
        for encoder,label in [('clip','CLIP_I16'),('dino','DINO_I16')]:
            path=out/f'{encoder}.npz'
            if not path.exists():continue
            with np.load(path) as f:
                v=f['global_embedding'];record[label]=float((v[1:17]@v[0]).mean())
                if encoder=='dino' and 'matched_source' in f['names']:
                    v=f['foreground_embedding'];record['ED_R']=float(1-v[-1]@v[0])
        if (out/'pixels.json').exists():record.update(read_json(out/'pixels.json'))
        rows.append(record)
    summary=[]
    metrics=['CLIP_I16','DINO_I16','ED_R','PSNR_dB','SSIM','LPIPS_alex','silhouette_IoU',
             'Width_APE_percent','Height_APE_percent','Box_extent_IoU']
    for dataset in sorted({r['dataset'] for r in registry_data['assets']}):
        for cohort in ['all_valid_exports','final_keep']:
            selected=[r for r in rows if r['dataset']==dataset and r['valid_16_views'] and
                      (cohort!='final_keep' or r['quality_status']=='keep')]
            values={}
            for key in metrics:
                values_for_key=[r[key] for r in selected if r.get(key) is not None]
                values[key]=dict(n=len(values_for_key),mean=float(np.mean(values_for_key)) if values_for_key else None)
            for encoder,label in [('fid','FID_2048'),('dino','FD_DINO')]:
                source,rendered=[],[]
                for row in selected:
                    path=OUT/row['obj_id']/f'{encoder}.npz'
                    if not path.is_file():continue
                    with np.load(path) as f:
                        v=f['global_embedding']
                        if v.ndim!=2 or len(v)<17 or not np.isfinite(v[:17]).all():
                            raise ValueError('Invalid distribution features: '+str(path))
                        if encoder=='dino' and not np.allclose(np.linalg.norm(v[:17],axis=1),1.,atol=1e-5,rtol=1e-5):
                            raise ValueError('FD-DINO requires L2-normalized final CLS: '+str(path))
                        source.append(v[0]);rendered.extend(v[1:17])
                values[label]=dict(n_identities=len(source),n_rendered=len(rendered),
                    feature_dimension=len(source[0]) if source else None,
                    feature_normalization='L2-normalized final CLS' if encoder=='dino' else 'unnormalized Inception pool3',
                    value=fid_value(np.asarray(source),np.asarray(rendered)))
            cohort_rows=[r for r in rows if r['dataset']==dataset and
                         (cohort!='final_keep' or r['quality_status']=='keep')]
            summary.append(dict(dataset=dataset,cohort=cohort,valid_queries=len(selected),
                registered_exports=len(cohort_rows),excluded_invalid_render=len(cohort_rows)-len(selected),
                missing_ply_at_summary=sum(not r['ply_present_at_summary'] for r in cohort_rows),
                valid_cached_render_without_ply=sum(not r['ply_present_at_summary'] for r in selected),
                metrics=values))
    registry_data['protocol']=dict(registry_data['protocol'],
        output_availability='Registered exports include historical outputs removed by user-authorized cleanup. Missing PLY counts describe current file presence only, not hash validation. A previously verified valid render cache may still support metrics after PLY deletion; this overlap is counted separately. Exclusions do not change production AOR query denominators. Missing outputs can bias the available-render cohort.')
    write_json(N/f'pilot_results/{RESULT_NAME}.json',dict(protocol=registry_data['protocol'],summary=summary,assets=rows))
    with (N/f'pilot_results/{RESULT_NAME}_summary.csv').open('w',newline='') as handle:
        accounting=['registered_exports','excluded_invalid_render','missing_ply_at_summary','valid_cached_render_without_ply']
        fields=['dataset','cohort','valid_queries',*accounting,*metrics,'FID_2048','FD_DINO','FID_identities']
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        writer.writerows(dict(dataset=r['dataset'],cohort=r['cohort'],valid_queries=r['valid_queries'],
            **{key:r[key] for key in accounting},
            **{key:r['metrics'][key]['mean'] for key in metrics},
            FID_2048=r['metrics']['FID_2048']['value'],FD_DINO=r['metrics']['FD_DINO']['value'],
            FID_identities=r['metrics']['FID_2048']['n_identities'])
            for r in summary)
    print(json.dumps(summary),flush=True)


async def pipeline(*,partial=False):
    document=registry(partial=partial);OUT.mkdir(parents=True,exist_ok=True)
    async def launch(python,args,gpu,name):
        gpu=await slots.acquire('render')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PATH=str(Path(python).parent)+':'+os.environ['PATH'],
            CPATH='/home/wmj/miniconda3/envs/splatad/include',MAX_JOBS='2',
            TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions'))
        process=None
        try:
            with (N/f'evaluation/{name}.log').open('a') as log:
                process=await asyncio.create_subprocess_exec(python,str(Path(__file__).resolve()),*args,env=env,
                    stdout=log,stderr=asyncio.subprocess.STDOUT)
                if await process.wait():raise RuntimeError(f'Metric stage failed: {name}')
        finally:
            if process is not None and process.returncode is None:
                process.terminate();await process.wait()
            slots.release(gpu)
    gpus=[int(v) for v in os.environ.get('ASSET_METRIC_GPUS','3,4,5,6,7').split(',')]
    if not gpus or len(set(gpus))!=len(gpus):raise ValueError('Provide distinct metric GPU indices')
    slots=GpuPool(gpus)
    await asyncio.gather(*(launch('/home/wmj/miniconda3/envs/sim/bin/python',
        ['render','--indices',*map(str,range(i,len(document['assets']),len(gpus)))],gpu,f'render_gpu{gpu}')
        for i,gpu in enumerate(gpus) if i<len(document['assets'])))
    feature_python='/mnt/ojc/miniconda3/envs/flux2_klein_harmonization_v2/bin/python'
    await asyncio.gather(*(launch(feature_python,['features','--encoder',encoder],gpus[i%len(gpus)],encoder)
                          for i,encoder in enumerate(['clip','dino','fid'])),
                        launch(str(R/'env_refine/bin/python'),['pixels'],gpus[3%len(gpus)],'pixels'))
    await launch(feature_python,['summarize'],gpus[-1],'summary')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['pipeline','render','features','pixels','summarize'])
    parser.add_argument('--indices',nargs='+',type=int);parser.add_argument('--encoder',choices=['clip','dino','fid'])
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    if args.action=='pipeline':asyncio.run(pipeline(partial=args.partial))
    elif args.action=='render':render(args.indices)
    elif args.action=='features':features(args.encoder)
    elif args.action=='pixels':pixels()
    else:summarize()
