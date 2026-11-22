import json,sys
from pathlib import Path
import numpy as np
from plyfile import PlyData
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'DriveLab'))
from asset_extraction.bank import publish
for name,packet in [('V05_photo_hybrid','V05_road_work'),('P01_photo_prior','P01_road_closed')]:
 r=W/'runs'/name;d=r/'curation';g=json.loads((r/'generation/generation.json').read_text());row=g['assets'][0];c=json.loads((d/'canonical.json').read_text());render=json.loads((d/'final_render/render.json').read_text());post=json.loads((d/'postprocess.json').read_text())
 assert render['ply_sha256']==c['sha256'],'Stale visual evidence'
 v=PlyData.read(c['canonical_path'])['vertex'].data;p=np.column_stack([v[k] for k in ['x','y','z']]);c['size_xyz']=np.ptp(p,axis=0).tolist()
 metric=c['metric_preview'];prior=name.startswith('P01')
 entry=dict(uid=row['obj_id']+'_'+name,category='traffic_sign',route=row['route'],status='review',quality_score=6,
 source_path=c['canonical_path'],canonical=c,scale=dict(height_m=metric['height_m'],width_m=metric['width_m'],depth_m=metric['depth_m'],confidence='low',source='Manufacturer frame dimensions, rear depth assumed' if prior else 'Explicit provisional total height 1.6 m; exact source variant unresolved'),
 defects=post['defects'],label=dict(sub_type=row['sign_metadata']['asset_subtype'],usable_as=['road_closure'] if prior else ['temporary_construction']),
 provenance=dict(source_obj_id=row['obj_id'],scene_id=row['scene_id'],source_image=row['image_path'],source_image_sha256=row['image_sha256'],source=row['source'],real_images=[],real_view_selection=None,prompts=row['prompts'],input_mode='source_photo_geometric_prior' if prior else 'source_photo_B0_support_hybrid',parent_raw_sha256=row['raw_sha256'],generation_signature=g['signature'],postprocess=post),
 quality=dict(reviewer='Codex source/front/back/side/orbit inspection',status='review',reason='Source text retained; geometry completion and scale uncertainty explicitly prevent keep admission',rendering=render,source_rgba=str(W/'inputs'/packet/'rgba.png'),source_mask=str(W/'inputs'/packet/'mask.png'),low_evidence=str(d/'low_render/evidence.jpg'),orbit_video=str(d/'orbit.mp4'),postprocess_receipt=str(d/'postprocess.json'),rectified_source_plate=str(d/'rectified_source_plate.png'),reviewed_views=['source and 3-background alpha','16 azimuths at 12 degrees','16 low views','source texture mapping chirality test']),
 **{k:row[k] for k in ['sign_metadata','sign_metadata_sha256','backend_policy']})
 state=dict(assets={entry['uid']:dict(status='review',entry=entry)})
 (d/'review_state.json').write_text(json.dumps(state,indent=2));added=publish(state,Path('/mnt/ojc/workplace1/asset_library'),dataset='emergency_traffic_signs',scene_id=row['scene_id']);(d/'published.json').write_text(json.dumps(added,indent=2));print(name,added[0]['package_path'])
