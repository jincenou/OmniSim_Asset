"""Freeze visually reviewed inputs before model output exists."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1]
SELECTION={
 'service_van':dict(train=[1,5,9,11],heldout=[7,10],role='nearby parked service van; moderate partial occlusion in early view'),
 'silver_sedan_03':dict(train=[1,4,7,10],heldout=[3,9],role='moving sedan; native laser speed about 16–18 m/s'),
 'white_sedan':dict(train=[0,2,4],heldout=[3],role='difficult low-resolution parked sedan; three usable real views, not padded to four'),
}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
cases=[]
for name,sel in SELECTION.items():
    out=ROOT/'inputs'/name
    views=json.loads((out/'view_candidates.json').read_text())
    inventory=json.loads((out/'associated_inventory.json').read_text())
    source=inventory['source'];root=Path(source['archive_root'])
    images=[]
    for split in ['train','heldout']:
        for i in sel[split]:
            rec=dict(views[i]);rec.update(split=split,sha256=sha(rec['path']),id=f'real_f{rec["frame_index"]:03}_c{rec["camera"]}')
            images.append(rec)
    for key,label in [('flux_front_copy','flux_end'),('flux_side_copy','flux_side')]:
        src=root/source[key];dest=out/(label+'.png');shutil.copy2(src,dest)
        images.append(dict(id=label,path=str(dest),source_path=str(src),sha256=sha(dest),source_type='flux',split='train',
            source_crop=str(root/source['source_copy']),camera=None,K=None,pose=None,
            label_note='Archived end prompt requested front; source observes rear for van/white sedan. No calibrated generated camera.',
            original_obj_id=source['obj_id']))
    real=[r for r in images if r['source_type']=='real' and r['split']=='train']
    # Shared metric scale comes only from real INPUT observations, never heldout.
    import numpy as np
    dims=np.median([r['projected_laser_box'][3:6] for r in real],axis=0).tolist()
    case=dict(case_id=name,original_obj_id=source['obj_id'],role=sel['role'],setup='metadata-assisted',
        source_manifest=str(root/'manifest.json'),selection_method='input-only geometry + visual occlusion review',
        images=images,metric_dimensions_lwh=dims,scale_source='median native laser boxes of training views only',
        identity=inventory['projection_audit'],flux_seed_provenance=source,
        flux_quality_note='visible instance/design drift retained as measured input limitation',
        heldout_policy='Never used for generation, refinement, alignment, size fitting or parameter choice')
    (out/'manifest.json').write_text(json.dumps(case,indent=2));cases.append(case)
(ROOT/'inputs/frozen_cases.json').write_text(json.dumps(cases,indent=2))
(ROOT/'inputs/experiment_matrix.json').write_text(json.dumps([
 dict(case_id=c['case_id'],variant=v,seed=42,status='pending',
      real_count=sum(i['source_type']=='real' and i['split']=='train' for i in c['images']) if v in ['R','H'] else 0,
      flux_count=2 if v in ['B0','F','H'] else 0)
 for c in cases for v in ['B0','R','F','H']],indent=2))
print('Frozen',len(cases),'identities and 12 planned variants')
