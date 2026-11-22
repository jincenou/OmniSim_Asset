from pathlib import Path
import os,sys,json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];REPO=R.parents[1]/'round2/moving61_runtime';sys.path.insert(0,str(REPO))
from gpu_guard import validate_gpu
from geometry_precision import build
from root_local_geometry import separate_horizontal_root,source_camera_maps
from fdanyone.skeleton import pipeline as skeleton
from fdanyone.motion.result import MotionResult
from fdanyone.assets import resolve_regressor

def probe():
 device=validate_gpu();stable=build(skeleton._body_geometry);results=[]
 for label in ['N01','N04','N06']:
  source=R/'samples'/label/'source';motion=MotionResult.load(source/'motion_pair/calibrated_moving');both={};geometries={}
  for name,function in [('original',skeleton._body_geometry),('stable_same_weights',stable)]:
   geometry=function(motion,resolve_regressor(None,model_dir=REPO/'models'),REPO/'third_party/GVHMR','cuda:0');local,maps,checks=separate_horizontal_root(geometry);geometries[name]=geometry
   try:_,error=source_camera_maps(geometry,local,maps['local_to_canonical']);both[name]={'passed':True,'rigid_error':error}
   except ValueError as error:both[name]={'passed':False,'failure':str(error)}
  a,b=geometries.values();camera=np.load(source/'inputs/cameras.npz')['K_input'];uvs=[]
  for g in [a,b]:
   q=np.einsum('tij,tkj->tki',camera,g.keypoints_incam);uvs.append(q[...,:2]/q[...,2:])
  results.append({'label':label,'same_body_vertices_exact':bool(np.array_equal(a.vertices_world,b.vertices_world)),'keypoint_world_max_difference':float(abs(a.keypoints_world-b.keypoints_world).max()),'keypoint_incam_max_difference':float(abs(a.keypoints_incam-b.keypoints_incam).max()),'source_projection_max_difference_px':float(abs(uvs[0]-uvs[1]).max()),'rigid_guard_unchanged_1e4':both})
  print(json.dumps(results[-1]),flush=True)
 record={'state':'probe_complete_no_geometry_tolerance_relaxed','device':device,'stable_function_sha256':stable.random8_precision_source_sha256,'change':'Accumulate same existing 640term vertex-regression products in float64,then cast to originalfloat32; no weight normalization or SMPL change','results':results};(R/'research/geometry_precision_probe.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':probe()
