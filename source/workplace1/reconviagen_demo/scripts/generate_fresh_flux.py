"""Reuse wmj flux._generate verbatim with a readable local 9B checkpoint."""
from pathlib import Path
import os,sys,json,time,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1]
for k,v in {'HF_HOME':'hf','TORCH_HOME':'torch','XDG_CACHE_HOME':'xdg','NUMBA_CACHE_DIR':'numba'}.items():os.environ[k]=str(ROOT/'cache'/v)
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,'/mnt/wmj/workspace4/pipeline')
import flux
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--item-index',type=int,required=True);p.add_argument('--manifest',type=Path,default=ROOT/'inputs/fresh_flux_manifest.json');a=p.parse_args()
out=ROOT/'inputs'/a.case;case=json.loads((out/'segmented_manifest.json').read_text())
assert not any(r['source_type']=='flux' for r in case['images'])
reference=next(r for r in case['images'] if r['id']==case['flux_reference_id']);assert reference['split']=='train'
rgba=Image.open(reference['rgba_path']).convert('RGBA');white=Image.new('RGBA',rgba.size,'white');white.alpha_composite(rgba);white.convert('RGB').save(out/'flux_reference.png')
handoff=a.manifest.resolve()
items=flux.load_items(handoff,out/'flux_state.json')
item=items[a.item_index]
assert item['obj_id']==a.case and item['category']=='car'
assert item['image_path']==str(out/'flux_reference.png')
prompts=item['prompts'];assert len(prompts)==2
task=dict(item_index=a.item_index,image_path=item['image_path'],prompts=prompts,output_paths=[str(out/'flux_end.png'),str(out/'flux_side.png')])
weight=Path('/mnt/ojc/models/flux2_klein/FLUX.2-klein-9B')
audit=dict(task=task,input_json_path=str(handoff),input_item=item,model_path=str(weight),model_index=json.loads((weight/'model_index.json').read_text()),config=vars(flux.CONFIG),seed_per_view=[a.item_index*1000+i for i in range(2)],source_script=flux.__file__,source_sha256=hashlib.sha256(Path(flux.__file__).read_bytes()).hexdigest(),prompt_source='Exact vehicle template from original rigid_5k_manifest.json',reference_source=reference['id'],reference_processing='SAM2 foreground alpha composited on white; original resize_reference_image',selection_source='New native Waymo track selection, not original detector/tracklet rerun',status='running')
(out/'flux_generation.json').write_text(json.dumps(audit,indent=2))
t=time.monotonic();flux.PIPE=Flux2KleinPipeline.from_pretrained(str(weight),torch_dtype=torch.bfloat16,local_files_only=True);flux.PIPE.enable_model_cpu_offload();flux.PIPE.set_progress_bar_config(disable=True)
audit['load_seconds']=time.monotonic()-t;torch.cuda.reset_peak_memory_stats();t=time.monotonic()
paths=flux._generate(task);torch.cuda.synchronize();audit.update(generation_seconds=time.monotonic()-t,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),status='succeeded')
for label,path in zip(['flux_end','flux_side'],paths):
 case['images'].append(dict(id=label,path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),source_type='flux',split='train',source_crop=reference['path'],camera=None,K=None,pose=None,rgba_path=str(out/(label+'_rgba.png')),mask_path=str(out/(label+'_mask.png')),generation_manifest=str(out/'flux_generation.json')))
(out/'segmented_manifest.json').write_text(json.dumps(case,indent=2))
# Preserve the immutable real selection and append newly generated images to the experiment manifest.
base=json.loads((out/'manifest.json').read_text());base['images']+=case['images'][-2:];(out/'manifest.json').write_text(json.dumps(base,indent=2))
(out/'flux_generation.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2),flush=True)
