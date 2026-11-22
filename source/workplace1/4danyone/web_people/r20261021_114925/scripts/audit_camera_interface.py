"""Read-only audit of the historical P31 G0 camera/coordinate path."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from safetensors import safe_open

W=Path(__file__).resolve().parents[1]
P=Path('/mnt/ojc/workplace1/4danyone')
S=P/'waymo/scene20/r20261019/scenes/092/P31'
sys.path[:0]=[str(P/'repo'),str(W/'research/baseline_evidence')]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    import torch
    torch.set_num_threads(4)
    from fdanyone.motion.result import MotionResult
    from fdanyone.skeleton.pipeline import _body_geometry
    from fdanyone.assets import resolve_regressor
    from geometry_precision import build
    from root_local_geometry import separate_horizontal_root,source_camera_maps
    cams=np.load(S/'input/source_cameras.npz')
    native=np.load(S/'input/native_cameras.npz')
    stored=np.load(S/'root_motion.npz')
    with safe_open(S/'motion/motion.safetensors',framework='numpy') as f:
        K=f.get_tensor('K_fullimg')
    K_error=float(np.abs(K-cams['K_input'].astype('float32')).max())
    K_original=np.linalg.inv(cams['undistorted_to_input'])@cams['K_input']
    calibration_path=Path('/mnt/public_data/waymo/val/092/intrinsics/1.txt')
    values=np.loadtxt(calibration_path)
    calibrated_K=np.array([[values[0],0,values[2]],[0,values[1],values[3]],[0,0,1]])
    calibration_error=float(np.abs(K_original-calibrated_K).max())
    assert calibration_error<1e-9
    native_original=np.linalg.inv(native['undistorted_to_input'])@native['K_input']
    rot=cams['R_world_to_cv_camera']
    relative=rot[1:]@np.swapaxes(rot[:-1],1,2)
    angles=np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1))
    motion=MotionResult.load(S/'motion')
    geometry=build(_body_geometry)(motion,resolve_regressor(None,model_dir=P/'repo/models'),P/'repo/third_party/GVHMR','cpu')
    local,maps,checks=separate_horizontal_root(geometry)
    source_map,error=source_camera_maps(geometry,local,maps['local_to_canonical'])
    maps['local_to_smpl_incam']=source_map
    map_errors={key:float(np.abs(value-stored[key]).max()) for key,value in maps.items()}
    xyz=np.einsum('tij,tnj->tni',K,geometry.keypoints_incam)
    uv=xyz[...,:2]/xyz[...,2:]
    projection_error=float(np.abs(uv-np.load(S/'motion/projection.npz')['keypoints_uv']).max())
    assert K_error==0 and max(map_errors.values())<1e-4 and projection_error<.01
    historical=P/'waymo/frontend_general_v1/validation12/downstream'
    native_equal=digest(W/'backend/scripts/train_grid.py')==digest(historical/'scripts/train_grid.py')
    continuation_equal=digest(W/'backend/scripts/train_continuation.py')==digest(historical/'scripts/train_continuation.py')
    plan=json.loads((W/'backend/PLAN.json').read_text())['training']
    assert native_equal and continuation_equal
    assert plan['train_views']==[i for i in range(24) if i not in (0,8,16)]
    assert plan['total_actual_updates']==sum(plan[k] for k in ('native_actual_updates','alpha_actual_updates','A_actual_updates'))==52999
    record=dict(state='historical_runtime_coordinate_evidence_verified_new_input_not_yet_run',sample=str(S),
        evidence_scope='Existing prediction tensors plus fresh CPU body/coordinate replay, not a fresh GVHMR inference',
        K_prediction_vs_camera_float32_max_abs=K_error,
        reconstructed_original_K_first=K_original[0].tolist(),
        dataset_intrinsics_path=str(calibration_path),dataset_intrinsics_sha256=digest(calibration_path),
        recovered_K_vs_dataset_intrinsics_max_abs=calibration_error,
        original_K_variation_max_abs=float(np.abs(K_original-K_original[0]).max()),
        native_vs_interpolated_original_K_max_abs=float(np.abs(native_original[0]-K_original[0]).max()),
        rotation_step_degrees_max=float(np.rad2deg(angles).max()),
        rotation_input='Code explicitly computes cam_angvel from saved nonconstant rotations; original cam_angvel tensor was not retained',
        camera_translation_consumed_by_GVHMR=False,
        native_camera_translation_present=True,
        root_checks=checks,root_map_replay_max_abs=map_errors,
        global_incam_rigid_recomposition_max_abs=error,stored_source_projection_max_abs_px=projection_error,
        repeated_Waymo_extrinsic_application='None in audited motion->canonical->root_local path; source map is same-body rigid recomposition, not dataset extrinsics',
        unverified=['True metric world trajectory','Human pose correctness','Rolling shutter recovery','Original inference cam_angvel tensor value beyond code/saved-rotation evidence'],
        full_A=dict(native_training_code_identical=native_equal,continuation_code_identical=continuation_equal,
            source=str(historical),protocol=plan,
            scope='Source/config audit before new training; runtime optimizer counts/LR/recovery assertions will be checked per new asset'),
        files={str(f):digest(f) for f in [S/'input/source_cameras.npz',S/'input/native_cameras.npz',S/'motion/motion.safetensors',S/'root_motion.npz',S/'motion/projection.npz']})
    (W/'research/camera_historical_audit.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record,indent=2,allow_nan=False))

if __name__=='__main__':main()
