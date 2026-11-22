"""Download one explicit public timm DINOv3 release, pinned and SHA checked."""
import os,json,hashlib
from dataclasses import asdict
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
ROOT=Path(__file__).resolve().parents[1]
repo='timm/vit_huge_plus_patch16_dinov3_qkvb.lvd1689m'
revision='2f1120d77def00163a5d553c2fdfabc1d8bc1037'
meta=HfApi().model_info(repo,revision=revision,files_metadata=True)
assert meta.sha==revision and not meta.gated
records=[]
for f in meta.siblings:
 if f.rfilename not in ['config.json','model.safetensors','README.md','LICENSE.md']:continue
 item=dict(rfilename=f.rfilename,size=f.size,lfs=asdict(f.lfs) if f.lfs else {})
 print('DOWNLOAD',f.rfilename,f.size,flush=True)
 path=Path(hf_hub_download(repo,f.rfilename,revision=revision,local_dir=ROOT/'cache'/repo.replace('/','--')))
 if path.stat().st_size!=f.size:raise ValueError('Size mismatch')
 if f.lfs:
  h=hashlib.sha256()
  with path.open('rb') as stream:
   for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
  if h.hexdigest()!=f.lfs.sha256:raise ValueError('SHA256 mismatch')
 records.append(item)
(ROOT/'docs/dinov3_weights.json').write_text(json.dumps(dict(repo=repo,revision=revision,files=records,path=str(ROOT/'cache'/repo.replace('/','--')),original_backbone='DINOv3 ViT-H+/16 distilled on LVD-1689M',implementation='timm qkvb variant; original zero QKV biases retained; RoPE periods truncated through bfloat16 as documented by timm to reproduce original frequencies',official_transformers_release='facebook/dinov3-vith16plus-pretrain-lvd1689m was inaccessible (403 gated); no DINOv2 substitution'),indent=2))
print('DINOv3 download verified',flush=True)
