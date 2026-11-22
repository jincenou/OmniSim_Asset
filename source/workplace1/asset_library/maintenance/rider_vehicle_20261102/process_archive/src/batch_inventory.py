"""Count unique assemblies and matching-version artifacts, never method variants."""
from pathlib import Path
import json,hashlib
W=Path(__file__).resolve().parents[1];root=W.parents[2];rows=[]
for r in sorted((W/'runs').iterdir()):
 if not (r/'flux_input.json').exists():continue
 item={'identity':r.name,'category':json.loads((r/'flux_input.json').read_text())[0]['category'],'successful_reconstruction_methods':[],'matching_insertions':[],'package':None,'accepted':False}
 for statepath in r.glob('*/generation.json'):
  state=json.loads(statepath.read_text())
  for a in state.get('assets',[]):
   if a.get('raw_path') and Path(a['raw_path']).is_file():item['successful_reconstruction_methods'].append(statepath.parent.name)
 canonical=r/'selected/canonical.ply'
 if canonical.exists():
  sha=hashlib.sha256(canonical.read_bytes()).hexdigest()
  for meta in r.glob('insertion_*/meta.json'):
   d=json.loads(meta.read_text())
   if d.get('asset_sha256')==sha:item['matching_insertions'].append(meta.parent.name)
 uid='rider_'+r.name if r.name.startswith('bicycle_nusc') else 'rider_web_'+r.name;p=root/'rigid'/item['category']/uid
 if (p/'package.json').exists():
  item['package']=str(p.relative_to(root));item['quality_status']=json.loads((p/'quality.json').read_text())['status'];item['accepted']=item['quality_status'] in {'demo_accepted','accepted'}
 rows.append(item)
out={'unique_reconstructed':sum(bool(x['successful_reconstruction_methods']) for x in rows),'complete_candidate_packages':sum(x['package'] is not None for x in rows),'accepted_count':sum(x['accepted'] for x in rows),'scope':'Artifacts only; runtime liveness is checked separately; candidate packages are not accepted assets.','assets':rows}
p=W/'reports/current_inventory.json';p.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='assets'}))
