"""Whole-frame generated-view consistency. This is not real-view accuracy."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image
import torch
import yaml
from evaluate_asset import Renderer,save,sha,scalar_metrics
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

R=Path(__file__).resolve().parents[1]

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--manifest',type=Path,default=R/'benchmark/manifest.json');args=p.parse_args()
    manifest=json.loads(args.manifest.read_text())
    job=next(x for x in manifest['jobs'] if x['id']==args.job)
    physical=int(os.environ['CUDA_VISIBLE_DEVICES'])
    if physical not in (4,5,6,7):raise ValueError('Unauthorized GPU')
    expected=subprocess.check_output(['nvidia-smi',f'--id={physical}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().removeprefix('GPU-')
    torch.cuda.init();actual=str(torch.cuda.get_device_properties(0).uuid)
    if actual!=expected or torch.cuda.device_count()!=1:raise RuntimeError('CUDA UUID mismatch')
    torch.set_num_threads(4);started=time.monotonic()
    out=R/'benchmark/results'/args.job;out.mkdir(parents=True,exist_ok=False)
    save(out/'device.json',{'physical_gpu':physical,'actual_cuda_uuid':actual,'pid':os.getpid()})
    checkpoint=Path(job['checkpoint']);data=Path(job['data'])
    if sha(checkpoint)!=job['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    model=Renderer(checkpoint,data);views=[0,8,16]
    # Checkpoint embeds optimizer/render settings only. The emitted run config
    # owns view splitting; do not infer missing fields in the checkpoint.
    cfg_path=checkpoint.parents[1]/'cfg.yml'
    cfg=yaml.unsafe_load(cfg_path.read_text())
    if sorted(cfg.get('validation_view_ids') or list(range(0,24,cfg['test_every'])))!=views:raise ValueError('Unexpected heldout views')
    if set(cfg.get('train_view_ids') or [v for v in range(24) if v%cfg['test_every']]) & set(views):raise ValueError('Training leakage')
    lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).cuda().eval()
    def tensor(array):return torch.from_numpy(np.ascontiguousarray(array)).permute(2,0,1)[None].cuda()
    def measure(a,b):
        s=float(ssim(a,b,data_range=1.,gaussian_kernel=True,sigma=1.5,kernel_size=11,k1=.01,k2=.03).item())
        l=float(lpips(a,b).item());lpips.reset()
        return s,l
    rows=[];inputs=[];selfcheck=None
    for view in views:
        for frame in range(121):
            path=data/'images'/f'cam{view:02d}'/f'{frame:06d}.png'
            target_rgba=np.asarray(Image.open(path).convert('RGBA'),dtype=np.float32)/255
            raw,alpha=model.render(view,frame)
            if raw.shape!=(640,352,3) or target_rgba.shape!=(640,352,4):raise ValueError('Protocol requires existing352x640 master, no resizing')
            target=target_rgba[...,:3]*target_rgba[...,3:]+1-target_rgba[...,3:]
            predicted=np.clip(raw+1-alpha[...,None],0,1)
            a,b=tensor(predicted),tensor(target)
            if selfcheck is None:
                equal_s,equal_l=measure(b,b)
                if abs(equal_s-1)>1e-5 or abs(equal_l)>1e-5:raise ValueError('Identity metric check failed')
                selfcheck={'identical_image_ssim':equal_s,'identical_image_lpips':equal_l}
            s,l=measure(a,b)
            row=scalar_metrics(predicted,target,alpha,target_rgba[...,3]>.5)
            row.update(view=view,frame=frame,seconds=frame/25,ssim=s,lpips=l)
            if not all(np.isfinite(v) for v in row.values()):raise ValueError('Nonfinite metric')
            rows.append(row);inputs.append({'view':view,'frame':frame,'sha256':sha(path)})
        print(json.dumps({'job':args.job,'completed_view':view,'images':len(rows)}),flush=True)
    keys=['full_psnr','ssim','lpips','foreground_psnr','silhouette_iou','outside_alpha_per_foreground_pixel']
    aggregate=lambda sample:{k:float(np.mean([s[k] for s in sample])) for k in keys}
    save(out/'metrics.json',{'state':'complete','job':job,'protocol_sha256':sha(R/'benchmark/protocol.json'),
        'reference':'generated heldout observations; not Waymo real novel-view ground truth',
        'aggregates':aggregate(rows),'per_view':{str(v):aggregate([x for x in rows if x['view']==v]) for v in views},
        'samples':rows,'inputs':inputs,'frames':121,'heldout_views':views,'evaluated_images':len(rows),
        'steps':model.state['step']+1,'gaussians':len(model.s['means']),'metric_selfcheck':selfcheck,
        'split_config':str(cfg_path),'split_config_sha256':sha(cfg_path),
        'versions':{x:importlib.metadata.version(x) for x in ['torch','torchmetrics','gsplat']},
        'seconds':time.monotonic()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'no_point_filtering':True,'visual_approval':False})

if __name__=='__main__':main()
