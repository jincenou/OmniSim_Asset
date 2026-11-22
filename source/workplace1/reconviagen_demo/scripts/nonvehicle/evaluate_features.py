from pathlib import Path
import sys,json,argparse,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_asset_metrics import Encoders,foreground
R=Path(__file__).resolve().parents[2]/'nonvehicle10'
p=argparse.ArgumentParser();p.add_argument('--encoder',choices=['clip','dino'],required=True);p.add_argument('--cases',nargs='+',required=True);a=p.parse_args();model=Encoders(a.encoder)
for cid in a.cases:
 case=json.loads((R/'inputs'/cid/'segmented_manifest.json').read_text());src=next(r for r in case['images'] if r['id']==case['flux_reference_id']);im,mask,crop=foreground(src['path'],src['mask_path']);sf=model.encode([im],[mask])['global_embedding'][0]
 for variant in ['B0','R','H','B0_real1']:
  out=R/'results'/cid/f'{variant}_seed42';render=out/'orbit'
  if not (render/'render_config.json').exists():continue
  images=[];masks=[]
  for i in range(16):
   im,mask,_=foreground(render/f'view_{i:02}.png',render/f'alpha_{i:02}.png');images.append(im);masks.append(mask)
  ff=[]
  for i in range(0,16,2):ff.append(model.encode(images[i:i+2],masks[i:i+2])['global_embedding'])
  sims=np.concatenate(ff)@sf;result=dict(case_id=cid,variant=variant,encoder=a.encoder,I16=float(sims.mean()),view_cosines=sims.tolist(),ED_R=None,ED_R_reason='No independently calibrated source-to-generated-object pose; no best-view selection',source_id=src['id'],source_sha256=src['sha256'],source_mask_sha256=hashlib.sha256(Path(src['mask_path']).read_bytes()).hexdigest(),source_crop=crop,protocol='docs/evaluation_protocol.json',model='OpenAI CLIP ViT-B/32' if a.encoder=='clip' else 'DINOv3 ViT-H+/16 qkvb; normalized CLS; float32; RoPE bf16->fp32',input_real_count=sum(r['source_type']=='real' for r in case['images']))
  (out/f'{a.encoder}_metrics.json').write_text(json.dumps(result,indent=2));print(cid,variant,a.encoder,result['I16'],flush=True)
