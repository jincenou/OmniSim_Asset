"""Run original DriveLab TRELLIS on both archived FLUX inputs (explicit B0)."""
from pathlib import Path
import sys,os,json,time,argparse,traceback,hashlib
SHARED=Path(__file__).resolve().parents[2]
ROOT=SHARED/"nonvehicle10"
os.environ.setdefault('ATTN_BACKEND','flash-attn');os.environ.setdefault('SPCONV_ALGO','native')
os.environ['TORCH_HOME']='/mnt/ojc/models/trellis_runtime/torch'
os.environ['NUMBA_CACHE_DIR']=str(SHARED/'cache/numba')
sys.path.insert(0,'/mnt/wmj/workspace4/pipeline')
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--seed',type=int,default=42)
p.add_argument('--real-count',type=int,choices=[0,1,4],default=0)
p.add_argument('--repeat-control',action='store_true');a=p.parse_args()
if a.repeat_control and a.real_count:raise ValueError('Repeat control is FLUX-only')
tag='B0' if a.real_count==0 else f'B0_real{a.real_count}'
if a.repeat_control:tag='B0_repeat'
out=ROOT/'results'/a.case/f'{tag}_seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
if (out/'raw.ply').exists():raise FileExistsError(out/'raw.ply')
try:
    import torch
    from PIL import Image
    from gen_gs import GsConfig,GsBatchGenerator
    GsBatchGenerator._install_optional_import_stubs()
    from trellis import models
    from trellis.pipelines import samplers
    from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
    # Restrict the original loader's DINO hub call to an existing local cache.
    original=torch.hub.load
    def local_load(repo,name,*args,**kwargs):
        if repo=='facebookresearch/dinov2':repo='/mnt/ojc/models/trellis_runtime/torch/hub/facebookresearch_dinov2_main';kwargs['source']='local'
        return original(repo,name,*args,**kwargs)
    torch.hub.load=local_load
    cfg=GsConfig(model_dir=Path('/mnt/ojc/models/TRELLIS-image-large-gaussian'))
    t=time.monotonic();pipe=GsBatchGenerator(cfg)._load_pipeline(models,samplers,TrellisImageTo3DPipeline);pipe.cuda();load=time.monotonic()-t
    case=json.loads((ROOT/'inputs'/a.case/'segmented_manifest.json').read_text())
    rows=[r for r in case['images'] if r['source_type']=='flux' and r['split']=='train'];assert len(rows)>=1
    real=[r for r in case['images'] if r['source_type']=='real' and r['split']=='train']
    if a.real_count==1:rows.append(next(r for r in real if r['id']==case['flux_reference_id']))
    elif a.real_count==4:
        if len(real)!=4:raise ValueError('Expected exactly four frozen real training views')
        rows.extend(real)
    heldout={r['sha256'] for r in case['images'] if r['split']=='heldout'}
    for r in rows:
        if r['sha256'] in heldout:raise ValueError('Heldout leakage')
        if hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Frozen input changed')
        if not Path(r['rgba_path']).is_file():raise FileNotFoundError(r['rgba_path'])
    (out/'input_manifest.json').write_text(json.dumps(dict(case_id=a.case,variant=tag,images=rows,
        input_order='FLUX pair then selected real training views',heldout_used=False),indent=2))
    images=[pipe.preprocess_image(Image.open(r['rgba_path'])) for r in rows]
    for i,im in enumerate(images):im.save(out/f'processed_{i:02}.png')
    torch.cuda.reset_peak_memory_stats();t=time.monotonic()
    outputs=(pipe.run(images[0],seed=a.seed,formats=['gaussian'],preprocess_image=False) if len(images)==1 else pipe.run_multi_image(images,seed=a.seed,formats=['gaussian'],mode='multidiffusion',preprocess_image=False))
    torch.cuda.synchronize();elapsed=time.monotonic()-t
    if (out/'raw.ply').exists():raise FileExistsError(out/'raw.ply')
    outputs['gaussian'][0].save_ply(str(out/'raw.ply'))
    meta=dict(case_id=a.case,variant=tag,seed=a.seed,backend='original DriveLab TRELLIS',images=[r['path'] for r in rows],
        load_seconds=load,inference_seconds=elapsed,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        gaussian_count=int(outputs['gaussian'][0].get_xyz.shape[0]),mode='single_image' if len(images)==1 else 'multidiffusion',
        real_count=a.real_count,flux_count=sum(r["source_type"]=="flux" for r in rows),heldout_used=False,
        sparse_structure_sampler_params=pipe.sparse_structure_sampler_params,
        slat_sampler_params=pipe.slat_sampler_params,
        input_weighting='Original multidiffusion: equal per-image predictions; no custom real/FLUX weighting',
        difference_from_legacy='Category-general original TRELLIS; no special planar-sign reconstruction',
        status='inference_succeeded_downstream_pending')
    (out/'inference.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
except Exception:
    (out/'failure.txt').write_text(traceback.format_exc());raise
