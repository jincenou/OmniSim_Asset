from pathlib import Path
import json
W=Path(__file__).resolve().parents[1];root=W.parents[2];rows=[]
for run in sorted((W/'runs').iterdir()):
 if not (run/'flux_input.json').exists():continue
 inp=json.loads((run/'flux_input.json').read_text())[0];category=inp['category'];uid='rider_'+run.name if run.name.startswith('bicycle_nusc') else 'rider_web_'+run.name;dest=root/'rigid'/category/uid
 if not (dest/'package.json').exists():continue
 q=json.loads((dest/'quality.json').read_text());rows.append(dict(uid=uid,category=category,placement=str((dest/'placement.json').relative_to(root)),status=q['status'],production_release=False))
out=dict(schema_version=1,count=len(rows),scope='Whole-assembly packages; status distinguishes accepted internal demos from pending candidates; no public release',assets=rows,preview='previews/rider_vehicle_batch_20261103/index.html',additional_previews=['previews/web_rider_expansion_20261103/index.html'],accepted_internal_demos=sum(r['status']=='demo_accepted' for r in rows))
p=root/'manifests/rider_vehicle_batch_20261103.json';tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(out,indent=2));tmp.replace(p);print('PACKAGES',len(rows),'ACCEPTED',out['accepted_internal_demos'])
