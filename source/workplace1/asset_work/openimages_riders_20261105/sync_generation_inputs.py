"""Publish a deduplicated snapshot of source-screened seated inputs to H production."""
import json,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'inputs_150_seated'
def sync():
 selected=[];seen=set()
 excluded=json.loads((ROOT/'manual_input_exclusions.json').read_text())
 for name in ['manifest.json','additional_manifest.json']:
  p=OUT/name
  if not p.exists():continue
  for a in json.loads(p.read_text())['inputs']:
   if a['source_id'] in excluded or a['source']['sha256'] in seen or len(selected)>=150:continue
   # Both independent source checks must exist; image bytes stay evidence-bound.
   for key in ['seated_review','assembly_review']:
    if not Path(a[key]).is_file():raise ValueError('Missing source review: '+a['source_id'])
   seen.add(a['source']['sha256']);selected.append(a)
 value=dict(status='source_screened',approval_kind='Independent seated-pose and assembly reviews; final 3D quality admission still required',target=150,count=len(selected),categories=dict(collections.Counter(a['category'] for a in selected)),updated=time.time(),inputs=selected)
 p=OUT/'generation_inputs.json';tmp=p.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(p)
 return value
if __name__=='__main__':print({k:v for k,v in sync().items() if k!='inputs'})
