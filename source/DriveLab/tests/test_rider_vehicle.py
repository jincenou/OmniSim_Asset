import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from asset_extraction.rider_vehicle import assembly,uses_h,prompt,union_masks,validate_reference,validate_flux_review
from asset_extraction.vision import _clean_mask
from asset_extraction.export import _needs_heading

@pytest.fixture
def combo():
 return dict(category='motorcycle',assembly=dict(asset_kind='rider_vehicle_rigid',fixed_pose=True,rider_count=1,
 vehicle_subtype='seated scooter',propulsion_label='unknown',propulsion_evidence=[],pose_usage='riding',
 pose_description='seated',rider_appearance='blue coat',vehicle_appearance='gray scooter',hand_handlebar_contact='both hands',
 seat_contact='seated',foot_support='floorboard',source_id='source',shot_id='shot',identity='person-bike-1',pose_cluster_id='pose1',
 internal_articulation=False,wheel_animation=False,anchor_review=dict(accepted=True,complete_assembly=True,exactly_one_rider=True,seated=True,source='test')))

def test_opt_in_only(combo):
 assert uses_h(combo)
 assert not uses_h(dict(category='motorcycle'))
 assert not uses_h(dict(category='bicycle'))
 assert uses_h(dict(category='car'))
 assert _needs_heading('motorcycle',assembly(combo))
 assert not _needs_heading('motorcycle')

@pytest.mark.parametrize('field,value',[('rider_count',2),('rider_count',True),('fixed_pose',False),('wheel_animation',True),('pose_usage','walking'),('propulsion_label','electric_confirmed')])
def test_invalid_contract(combo,field,value):
 combo['assembly'][field]=value
 with pytest.raises(ValueError):assembly(combo)

def test_union_preserves_disconnected_support_and_holes():
 a=np.zeros((30,30),bool);b=a.copy();a[2:8,2:8]=True;b[15:25,15:25]=True;b[18:22,18:22]=False
 m=union_masks(a,b);preserved,_,_=_clean_mask(m,[0,0,30,30],preserve_components=True)
 assert np.array_equal(m,preserved)
 assert not preserved[19,19] and preserved[3,3]
 largest,_,_=_clean_mask(m,[0,0,30,30]);assert not largest[3,3]
 with pytest.raises(ValueError):union_masks(a,np.zeros_like(a))

def test_prompts_and_deepcopy(combo):
 a=assembly(combo);a['identity']='changed';assert assembly(combo)['identity']=='person-bike-1'
 p=prompt(combo);assert 'blue coat' in p and 'exactly one rider' in p and 'foot support' in p

def test_reference_pose_gate(combo):
 r=dict(identity='person-bike-1',pose_cluster_id='pose1',assembly_review={k:True for k in ['accepted','same_identity','same_pose','complete_assembly','complementary_view']})
 r['assembly_review']['source']='test';validate_reference(combo,r)
 r['pose_cluster_id']='pose2'
 with pytest.raises(ValueError):validate_reference(combo,r)

def test_flux_must_be_reviewed_by_exact_hash(combo):
 f=dict(images=[dict(sha256='abc')])
 with pytest.raises(ValueError):validate_flux_review(combo,f)
 combo['flux_assembly_reviews']=[dict(sha256='abc',accepted=True,complete_assembly=True,same_pose=True,same_identity=True,contacts_preserved=True,source='test')]
 validate_flux_review(combo,f)
 f['images'][0]['sha256']='changed'
 with pytest.raises(ValueError):validate_flux_review(combo,f)

def test_routes_and_jobs_preserve_and_hash_metadata(combo,tmp_path):
 from asset_extraction.generation import PATH_KEYS,prepare_jobs,sha256
 from nuscenes.prepare_routes import build_routes
 from asset_extraction.io_utils import write_json
 (tmp_path/'trellis/pipelines').mkdir(parents=True)
 for f in ['trellis/pipelines/trellis_image_to_3d.py','birefnet-general-lite.onnx','source.png']:(tmp_path/f).write_text('test')
 conf={k:str(tmp_path) for k in PATH_KEYS};conf.update(seed=42,segmentation_model=str(tmp_path/'birefnet-general-lite.onnx'))
 row=dict(scene_id='scene',obj_id='obj',category='motorcycle',image_path=str(tmp_path/'source.png'),prompts=[prompt(combo)])
 handoff=tmp_path/'handoff.json';routes=tmp_path/'routes.json';write_json(handoff,[row])
 source=dict(asset_id='obj',**combo,source=dict(identity='person-bike-1'),real_images=[],real_view_selection={})
 write_json(tmp_path/'scenes/scene/logs/export_items_full.json',[source]);write_json(routes,build_routes(handoff,tmp_path))
 first=prepare_jobs(handoff,routes,conf)
 assert first['assets'][0]['assembly']==combo['assembly'] and first['assets'][0]['route']=='H'
 assert set(json.loads(handoff.read_text())[0])=={'scene_id','obj_id','category','image_path','prompts'}
 data=json.loads(routes.read_text());data['assets'][0]['assembly']['vehicle_appearance']='red';write_json(routes,data)
 assert prepare_jobs(handoff,routes,conf)['signature']!=first['signature']
 data['assets'][0]['route']='B0';write_json(routes,data)
 with pytest.raises(ValueError,match='mismatch'):prepare_jobs(handoff,routes,conf)
