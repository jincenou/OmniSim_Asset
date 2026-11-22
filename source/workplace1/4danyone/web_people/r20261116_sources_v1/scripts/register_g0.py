"""Register actual produced web G0 products, without granting input approval."""
from pathlib import Path
import sys,json
import numpy as np
from prepare_person import ROOT,selected,save,sha
ident=sys.argv[1];choice=selected(ident);dest=ROOT/'inputs'/ident
front=ROOT/'frontend'/ident;motion=dest/'motion_pair/static_estimated'
for path in [front/'preflight/conditioning/cameras.json',front/'preflight/root_motion.npz',motion/'motion.safetensors']:
 assert path.is_file(),path
p=np.load(dest/'research/projection.npz')
coco=('nose','left-eye','right-eye','left-ear','right-ear','left-shoulder','right-shoulder','left-elbow','right-elbow','left-wrist','right-wrist','left-hip','right-hip','left-knee','right-knee','left-ankle','right-ankle')
sys.path.insert(0,'/mnt/ojc/workplace1/4danyone/repo')
from fdanyone.skeleton.keypoints import KEYPOINT_NAMES
dist=np.linalg.norm(p['skeleton_crop_uv'][:,[KEYPOINT_NAMES.index(n) for n in coco]]-p['observed_crop'][...,:2],axis=-1)
np.savez_compressed(dest/'research/diagnostic_projection.npz',observed_coco17=p['observed_crop'],detector_distance_px=dist,source_boxes=p['crop_boxes'])
np.savez_compressed(dest/'inputs/cameras.npz',K_input=p['K_crop'],K_original=p['K_original'],source_to_crop=p['A_t'],
 R_world_to_cv_camera=np.repeat(np.eye(3)[None],121,axis=0),calibrated=np.array(False),camera_model=np.array('estimated_K_approximately_static_source'))
save(dest/'native.json',dict(source=choice,source_provenance=str(ROOT/'sources'/f'{choice["source_id"]}_download.json'),
 source_pts=str(ROOT/'reviews'/choice['source_id']/'source_pts.json'),native_RGB_is_generated=False,calibration='estimated; not measured',
 model_time='i/25 for i=0..120; nearest original PTS; no interpolation or duplicate frames'))
registry=json.loads((ROOT/'config/cases.json').read_text());assert ident not in registry
registry[ident]=dict(prepared_frames=121,raw_input=str(dest/'inputs/main.mp4'),conditioning=str(front/'preflight/conditioning'),motion=str(motion),
 root_motion=str(front/'preflight/root_motion.npz'),source_dataset='web',input_packet=str(dest),projection=str(dest/'research/diagnostic_projection.npz'),
 input_boundary='Reviewed real web video; original-frame GVHMR then crop-K transformation, real PTS, predicted alpha, root-local 24-view conditions',
 selection=dict(id=ident,identity_key=choice['source_id']+'_'+str(choice['track_id']),package=str(dest),source=choice['source_path'],pose=str(dest/'pose_preview.mp4'),
 scene=choice['source_id'],rank_score=0,qualitative_only=True,upstream_example=False),
 manifest=str(ROOT/'data'/ident/'fdanyone.json'),config=str(ROOT/'config'/f'{ident}.json'),
 input_review=dict(allow_generation=False,accessory='no carried accessory visible; input review pending',input_sha256=sha(dest/'inputs/main.mp4')))
save(ROOT/'config/cases.json',registry)
print(ident,'registered; generation approval remains false')
