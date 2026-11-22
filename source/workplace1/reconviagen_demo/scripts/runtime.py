"""Headless Gaussian inference with pinned, strictly loaded official models."""
from pathlib import Path
import json, os, sys, time

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'vendor/ReconViaGen'
os.environ.setdefault('ATTN_BACKEND','flash-attn')
os.environ.setdefault('SPCONV_ALGO','native')
os.environ.setdefault('TORCH_HOME','/mnt/ojc/models/trellis_runtime/torch')
os.environ.setdefault('HF_HOME',str(ROOT/'cache/hf'))
os.environ.setdefault('HF_MODULES_CACHE',str(ROOT/'cache/hf_modules'))
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'cache/numba'))
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(REPO/'wheels/vggt'))

def load_pipeline(include_mesh=False):
    import torch
    from safetensors.torch import load_file
    from torchvision import transforms
    from trellis import models
    from trellis.pipelines import TrellisVGGTTo3DPipeline, samplers
    from trellis.pipelines.base import Pipeline
    from wheels.vggt.vggt.models.vggt import VGGT
    checkpoint=ROOT/'cache/Stable-X--trellis-vggt-v0-2'
    cfg=json.loads((checkpoint/'pipeline.json').read_text())['args']
    modules={};load_audit={}
    for key,value in cfg['models'].items():
        if key=='slat_decoder_rf' or (key=='slat_decoder_mesh' and not include_mesh):continue
        path=checkpoint/value
        mc=json.loads(path.with_suffix('.json').read_text())
        model=getattr(models,mc['name'])(**mc['args'])
        state=load_file(str(path.with_suffix('.safetensors')))
        # The pinned official loader uses strict=False. This exact checkpoint
        # retains eight fuse_blocks tensors whose module is commented out in
        # the pinned forward implementation; ignore only this audited set.
        allowed={f'fuse_blocks.{i}.0.{part}' for i in range(4) for part in ('weight','bias')} if key=='slat_vggt_cond' else set()
        extra=set(state)-set(model.state_dict())
        if extra-allowed:raise ValueError(f'Unexpected checkpoint keys for {key}: {sorted(extra-allowed)}')
        for name in extra:state.pop(name)
        model.load_state_dict(state,strict=True)
        load_audit[key]={'ignored_inactive_keys':sorted(extra),'all_active_keys_strict':True}
        modules[key]=model.eval()
        print('LOADED',key,flush=True)
    pipeline=TrellisVGGTTo3DPipeline()
    Pipeline.__init__(pipeline,modules)
    pipeline._pretrained_args=cfg
    pipeline.model_load_audit=load_audit
    pipeline._device=torch.device('cuda')
    pipeline.low_vram=True
    pipeline.VGGT_dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0]>=8 else torch.float16
    vc=ROOT/'cache/Stable-X--vggt-object-v0-1'
    vggt=VGGT(**json.loads((vc/'config.json').read_text()))
    vggt.load_state_dict(load_file(str(vc/'model.safetensors')),strict=True)
    del vggt.depth_head
    del vggt.track_head
    pipeline.VGGT_model=vggt.eval()
    for name in ['sparse_structure','slat']:
        spec=cfg[name+'_sampler']
        setattr(pipeline,name+'_sampler',getattr(samplers,spec['name'])(**spec['args']))
        setattr(pipeline,name+'_sampler_params',spec['params'])
    pipeline.slat_normalization=cfg['slat_normalization']
    # Local torch.hub source and existing DINO weights; never mutate this cache.
    pipeline._init_image_cond_model(cfg['image_cond_model'])
    # RGBA inputs have already been segmented. BiRefNet and DreamSim are not
    # invoked in base Gaussian inference; full refinement uses a separate entry.
    return pipeline

def inference(images, output_dir, seed=42):
    import torch
    from PIL import Image
    from preprocessing import preprocess_rgba
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    if (output_dir/'raw.ply').exists():raise FileExistsError(output_dir/'raw.ply')
    prepared=[]; records=[]
    for i,path in enumerate(images):
        image=Image.open(path)
        processed,record=preprocess_rgba(image)
        target=output_dir/f'processed_{i:02}.png';processed.save(target)
        prepared.append(processed);records.append(dict(source=str(path),processed=str(target),**record))
    (output_dir/'preprocessing.json').write_text(json.dumps(records,indent=2))
    t=time.monotonic();pipeline=load_pipeline();load_seconds=time.monotonic()-t
    torch.cuda.reset_peak_memory_stats();t=time.monotonic()
    with torch.no_grad():
        outputs,coords,noise=pipeline.run(prepared,seed=seed,formats=['gaussian'],preprocess_image=False)
    torch.cuda.synchronize()
    elapsed=time.monotonic()-t
    outputs['gaussian'][0].save_ply(str(output_dir/'raw.ply'))
    meta=dict(seed=seed,load_seconds=load_seconds,inference_seconds=elapsed,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        gaussian_count=int(outputs['gaussian'][0].get_xyz.shape[0]),formats=['gaussian'],low_vram=True,
        ss_sampler=pipeline.sparse_structure_sampler_params,slat_sampler=pipeline.slat_sampler_params,
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),model_load_audit=pipeline.model_load_audit,
        status='inference_succeeded_downstream_pending')
    (output_dir/'inference.json').write_text(json.dumps(meta,indent=2))
    return meta

if __name__=='__main__':
    import argparse,traceback
    p=argparse.ArgumentParser();p.add_argument('--images',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    try: print(json.dumps(inference(a.images,a.output,a.seed),indent=2))
    except Exception:
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
        (out/'failure.txt').write_text(traceback.format_exc());raise
