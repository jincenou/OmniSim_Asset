"""Replay cached unmasked evidence, record native stages without changing them.

CPU replay is intentional: current eight GPUs generate Base references. Numerical
agreement with the saved CUDA run is measured, never assumed bitwise identical.
"""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
import torch
from safetensors.torch import save_file,load_file
ROOT=Path(__file__).resolve().parent
PRIOR=ROOT.parent/'frontend_quality_20261020'
REPO=ROOT.parents[1]/'repo'
sys.path.insert(0,str(REPO))
from fdanyone.motion import gvhmr

def main():
    parser=argparse.ArgumentParser();parser.add_argument('sample');args=parser.parse_args()
    torch.set_num_threads(2)
    src=PRIOR/'motion_unmasked'/args.sample
    out=ROOT/'stage_audit'/args.sample;out.mkdir(parents=True,exist_ok=False)
    preprocess=src/'motion_pair/runtime/calibrated/preprocess'
    cameras=np.load(src/'inputs/cameras.npz')
    recorded={}
    def capture(name,outputs):
        for group in ['pred_smpl_params_incam','pred_smpl_params_global']:
            for key,val in outputs[group].items():recorded[f'{name}.{group}.{key}']=val.detach().cpu().clone()[0]
    with gvhmr.gvhmr_imports(REPO/'third_party/GVHMR'),gvhmr._legacy_checkpoint_loading():
        gvhmr.install_pytorch3d_compat();gvhmr._register_inference_store()
        import hydra
        from hydra import compose,initialize_config_module
        from hmr4d.utils.geo.hmr_cam import normalize_kp2d
        from hmr4d.utils.geo_transform import compute_cam_angvel
        import hmr4d.model.gvhmr.pipeline.gvhmr_pipeline as impl
        with initialize_config_module(version_base='1.3',config_module='hmr4d.configs'):
            cfg=compose(config_name='demo',overrides=['static_cam=false','verbose=false','use_dpvo=false'])
        model=hydra.utils.instantiate(cfg.model,_recursive_=False).eval().cpu()
        checkpoint_path=Path(cfg.ckpt_path).resolve()
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['state_dict'],strict=True);del checkpoint
        bbx=torch.load(preprocess/'bbx.pt',map_location='cpu',weights_only=True)['bbx_xys']
        kp=torch.load(preprocess/'vitpose.pt',map_location='cpu',weights_only=True)
        feat=torch.load(preprocess/'vit_features.pt',map_location='cpu',weights_only=True)
        K=torch.from_numpy(cameras['K_input']).float()
        batch={'length':torch.tensor([len(kp)]),'obs':normalize_kp2d(kp,bbx)[None],
               'bbx_xys':bbx[None],'K_fullimg':K[None],
               'cam_angvel':compute_cam_angvel(torch.from_numpy(cameras['R_world_to_cv_camera']).float())[None],
               'f_imgseq':feat[None]}
        original_contact=impl.pp_static_joint;original_ik=impl.process_ik
        def contact(outputs,endecoder):
            capture('pre_contact',outputs)
            recorded['contact_logits']=outputs['static_conf_logits'].detach().cpu().clone()[0]
            value=original_contact(outputs,endecoder)
            recorded['post_contact_transl']=value.detach().cpu().clone()[0]
            return value
        def ik(outputs,endecoder):
            capture('pre_ik',outputs)
            value=original_ik(outputs,endecoder)
            recorded['post_ik_body_pose']=value.detach().cpu().clone()[0]
            return value
        impl.pp_static_joint=contact;impl.process_ik=ik
        try:
            with torch.inference_mode():result=model.pipeline.forward(batch,train=False,postproc=True,static_cam=False)
        finally:
            impl.pp_static_joint=original_contact;impl.process_ik=original_ik
        capture('final',result)
    old=load_file(str(src/'motion_pair/calibrated_moving/motion.safetensors'))
    agreement={}
    for group in ['incam','global']:
        for key in ['body_pose','transl','global_orient','betas']:
            a=recorded[f'final.pred_smpl_params_{group}.{key}'];b=old[f'smpl_params_{group}.{key}']
            agreement[f'{group}.{key}']=float((a-b).abs().max())
    save_file({k:v.contiguous() for k,v in recorded.items()},str(out/'stages.safetensors'))
    report={'state':'native_cpu_stages_recorded','sample':args.sample,'source_variant':'unmasked_diagnostic',
      'strict_weight_load':True,'static_cam':False,'postproc':True,'contact_and_ik_unchanged':True,
      'cpu_vs_previous_cuda_max_abs':agreement,
      'scope':'Stage diagnosis only, not a repaired motion; CPU and GPU numerical paths may differ.',
      'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}
    (out/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
