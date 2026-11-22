from pathlib import Path
import sys,json,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'short_runtime'))
from fdanyone.model.conditioning import pose_feature_shape,POSE_FEATURE_SHAPE,PoseFeatureBank
from fdanyone.errors import FourDAnyoneError
from fdanyone.vendor.diffsynth.models.wan_video_pose_encoder import PoseEncoder
from fdanyone.vendor.diffsynth.models.wan_video_dit import MODEL_DIM
assert MODEL_DIM==3072
with torch.device('meta'):
 encoder=PoseEncoder(out_dim=MODEL_DIM,in_channels=3);rows=[]
 for frames in [21,61,121]:
  expected=pose_feature_shape(frames,1280,704);output=encoder(torch.empty((1,3,frames+encoder.temporal_prefix,1280,704)))
  assert tuple(output.shape)==(1,*expected),(frames,output.shape,expected)
  rows.append({'frames':frames,'actual_meta_pose_output':list(output.shape),'expected':list(expected)})
invalid=[]
for frames in [0,20,22,122,125]:
 try:pose_feature_shape(frames,1280,704)
 except FourDAnyoneError:invalid.append(frames)
 else:raise AssertionError(frames)
features=torch.zeros((2,*POSE_FEATURE_SHAPE),dtype=torch.bfloat16);null=torch.zeros((1,*POSE_FEATURE_SHAPE),dtype=torch.bfloat16);features[1].fill_(1)
bank=PoseFeatureBank(features,null);dest=bank.allocate_group(2,'cpu');bank.copy_group((1,0),dest)
assert dest[0].eq(1).all() and dest[1].eq(0).all()
report={'state':'CPU_contract_checks_passed_GPU_pending','cases':rows,'invalid_lengths_rejected':invalid,'feature_bank_copy_reorder':'passed','weights_loaded':False,'scope':'Actual3072-channel PoseEncoder meta shape and21-frame feature bank ownership/reorder checks.121 expectedshape unchanged; not full121 numerical end-to-end equivalence.','initial_test_correction':'First manual meta test used5120 defaultconstructor; corrected to3072 matching actual loader. No model code changed to pass that test.'}
(R/'short_temporal_preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
