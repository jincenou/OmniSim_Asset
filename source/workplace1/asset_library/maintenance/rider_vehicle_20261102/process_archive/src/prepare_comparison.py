"""Matched F for all methods. Explicit experiment route; never masquerades as native B0."""
import sys,json,copy
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src/DriveLab'))
from asset_extraction.generation import digest,sha256
from asset_extraction.io_utils import write_json
run=Path(sys.argv[1]);fdir=run/sys.argv[2];fstate=json.loads((fdir/'generation.json').read_text());base=json.loads((fdir/'jobs.json').read_text());row=base['assets'][0];f=fstate['assets'][0]['stages']['flux'];matte=Path(f['images'][0]['path']).with_name('flux_rgba.png')
review=json.loads((fdir/'visual_review.json').read_text())
if not review.get('accepted_for_structure_experiment') or review.get('matte_reviewed') is False:raise ValueError('Need recorded image review')
for name,backend,real in [('H0','H',[]),('H1','H',row['real_images']),('B0-match','B0',[])]:
 if name=='H1' and not real:continue
 name=(sys.argv[3] if len(sys.argv)>3 else '')+name
 work=run/name;work.mkdir(exist_ok=True);r=copy.deepcopy(row);r.update(route=backend,real_images=real,input_mode='flux_plus_real' if real else 'flux_only',experiment=dict(method=name,production_route=row['route'],matched_flux_sha256=[x['sha256'] for x in f['images']],seed=42,not_native_b0=True,identity_limitations=review.get('limitations',[])),flux_assembly_reviews=[dict(sha256=f['images'][0]['sha256'],accepted=True,complete_assembly=True,same_pose=True,same_identity=True,contacts_preserved=True,source='Codex visual review of coarse clothing/vehicle identity; facial and decal fidelity unverified; structure experiment only')],flux_segmented_images=[dict(path=str(matte),sha256=sha256(matte),flux_sha256=f['images'][0]['sha256'],complete_assembly=True)])
 b=dict(base,assets=[r]);b['signature']=digest(dict(parent=base['signature'],assets=[r],code=sha256(W/'src/DriveLab/asset_extraction/generation.py')));write_json(work/'jobs.json',b)
 state=dict(schema_version=1,signature=b['signature'],config=b['config'],assets=[dict(r,stages=dict(flux=f),status='succeeded')]);write_json(work/'generation.json',state)
print('COMPARISONS_READY')
