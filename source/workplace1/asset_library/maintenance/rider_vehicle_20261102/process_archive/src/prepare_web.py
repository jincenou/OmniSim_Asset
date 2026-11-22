"""Prepare reviewed native video shot as one assembly; adjacent frames never pad k."""
import json,sys
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src/DriveLab'))
from asset_extraction.io_utils import write_json
from asset_extraction.rider_vehicle import prompt
from asset_extraction.generation import prepare_jobs
from nuscenes.prepare_routes import build_routes
name=sys.argv[1];p=W/'packets'/name;x=json.loads((p/'segmentation.json').read_text());anchor=x['observations'][0]
attrs=json.loads((p/'attributes.json').read_text()); vid=attrs['video_id']; source_meta=json.loads((W/'sources'/name/'source.json').read_text()); appearance=attrs['rider_appearance']; vehicle=attrs['vehicle_appearance']
a=dict(asset_kind='rider_vehicle_rigid',fixed_pose=True,rider_count=1,vehicle_subtype='seated electric scooter',propulsion_label='electric_confirmed',propulsion_evidence=[vehicle+'; electric vehicle identified by the original video description at '+source_meta['page']],pose_usage='riding',pose_description='upright seated, both hands holding grips, bent knees and feet on floorboard',rider_appearance=appearance,vehicle_appearance=vehicle,hand_handlebar_contact='both hands on grips',seat_contact='hips resting on saddle',foot_support='both feet on floorboard',source_id=f'golem_{vid}',shot_id=anchor['source_observation']['shot_id'],identity=name,pose_cluster_id='seated_riding_01',internal_articulation=False,wheel_animation=False,anchor_review=dict(accepted=True,complete_assembly=True,exactly_one_rider=True,seated=True,source='Codex visual source and SAM2 inspection; pose stable within selected shot'))
row=dict(scene_id=f'golem_{vid}',obj_id=name,category='motorcycle',image_path=anchor['anchor_path']);row['prompts']=[prompt(dict(row,assembly=a))+' Preserve the source helmet style. Do not invent readable lettering. Keep the vehicle mechanically plausible and preserve both wheels.']
run=W/'runs'/name;run.mkdir(exist_ok=True)
source=dict(asset_id=name,category='motorcycle',assembly=a,source=dict(url=source_meta['page'],download_url=source_meta['download_url'],license='No explicit reuse license found; public manufacturer promotional film, internal research demo only, not licensed for redistribution',author='Original review or manufacturer film hosted by Golem; see source URL',vehicle_track_id=name+':vehicle',rider_track_id=name+':rider',identity_binding='visually reviewed single continuous shot; no cross-shot identity merge',anchor=anchor['source_observation']),real_images=[],real_view_selection=dict(real_count=0,preferred_count=[3,4],reason='Short constant-view shot; adjacent frames are near duplicates, not complementary views',duplicate_padding=False),vlm=None)
write_json(run/f'scenes/golem_{vid}/logs/export_items_full.json',[source]);write_json(run/'flux_input.json',[row]);routes=build_routes(run/'flux_input.json',run);write_json(run/'asset_routes.json',routes)
b=prepare_jobs(run/'flux_input.json',run/'asset_routes.json',json.loads((W/'configs/generation.json').read_text()));f=run/'F';f.mkdir(exist_ok=True);write_json(f/'jobs.json',b);write_json(p/'assembly.json',a);print(f)
