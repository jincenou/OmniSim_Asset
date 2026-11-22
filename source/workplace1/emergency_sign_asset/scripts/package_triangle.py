import json,sys
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'DriveLab'))
from asset_extraction.bank import publish
r=W/'runs/V01_B0';d=r/'curation';generation=json.loads((r/'generation/generation.json').read_text());row=generation['assets'][0];c=json.loads((d/'canonical.json').read_text());render=json.loads((d/'final_render/render.json').read_text())
entry=dict(uid=row['obj_id'],category='traffic_sign',route='B0',status='review',quality_score=6,
 source_path=c['canonical_path'],canonical=c,scale=dict(height_m=.42037,width_m=.43815,depth_m=None,confidence='medium',source='manufacturer diagram in video V01; depth unobserved'),
 defects=['Unobserved rear completed as second red/orange face; manufacturer geometry not verified','Reconstructed width at calibrated height is 0.415 m versus source 0.43815 m (~5.3% narrow)','Small embossed markings not recovered'],
 label=dict(sub_type='weighted_base_emergency_triangle',usable_as=['vehicle_breakdown','accident_warning']),
 provenance=dict(source_obj_id=row['obj_id'],scene_id=row['scene_id'],source_image=row['image_path'],source_image_sha256=row['image_sha256'],source=row['source'],real_images=[],real_view_selection=None,prompts=row['prompts'],input_mode='flux_only',raw_sha256=row['raw_sha256'],generation_signature=generation['signature']),
 quality=dict(reviewer='Codex visual inspection',status='review',reason='Front/opening/base retained; rear and thickness are generative completion, not confirmed geometry',rendering=render,scale_source='single deployed unit manufacturer diagram; no depth calibration',
 source_rgba=str(W/'inputs/V01_triangle/rgba.png'),source_mask=str(W/'inputs/V01_triangle/mask.png'),flux_rgba=str(r/'generation/asset_00000/segmented/rgba_reviewed.png'),low_evidence=str(d/'low_render/evidence.jpg'),orbit_video=str(d/'orbit.mp4'),
 reviewed_views=['16 azimuths at elevation 12 deg','16 azimuths at elevation 0 deg','native source and source mask','FLUX and reviewed alpha']),
 **{k:row[k] for k in ['sign_metadata','sign_metadata_sha256','backend_policy']})
state=dict(assets={row['obj_id']:dict(status='review',entry=entry)})
(d/'review_state.json').write_text(json.dumps(state,indent=2))
added=publish(state,Path('/mnt/ojc/workplace1/asset_library'),dataset='emergency_traffic_signs',scene_id=row['scene_id'])
(d/'published.json').write_text(json.dumps(added,indent=2));print(added[0]['package_path'])
