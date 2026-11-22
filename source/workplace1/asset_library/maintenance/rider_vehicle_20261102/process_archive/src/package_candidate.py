"""Portable candidate package. Does not publish to production/release catalog."""
from pathlib import Path
import sys,json,shutil,hashlib,os
import numpy as np
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src/DriveLab'))
from asset_extraction.gaussians import load_rigid
name,mode=sys.argv[1:];run=W/'runs'/name;sel=json.loads((run/'selected/selected.json').read_text());r=sel['row'];ins=run/('insertion_'+mode)
meta=json.loads((ins/'meta.json').read_text());assert meta['n_frames']>=24
load=json.loads((run/'selected/load_validation.json').read_text());assert load['passed']
assert hashlib.sha256((run/'selected/canonical.ply').read_bytes()).hexdigest()==meta['asset_sha256']
assert hashlib.sha256((W/'configs'/('scene0103_'+mode+'.npz')).read_bytes()).hexdigest()==meta['trajectory_sha256']
root=Path('/mnt/ojc/workplace1/asset_library');dest=root/'rigid'/r['category']/('rider_web_'+name if name!='bicycle_nusc_780be296' else 'rider_'+name)
if dest.exists():
 if ((dest/'package.json').exists() and hashlib.sha256((dest/'asset.ply').read_bytes()).hexdigest()==meta['asset_sha256']
     and (dest/'insertion'/mode/'meta.json').exists()
     and json.loads((dest/'insertion'/mode/'meta.json').read_text()).get('trajectory_sha256')==meta['trajectory_sha256']):
  print('PACKAGE_CACHED',dest);sys.exit(0)
 raise FileExistsError(dest)
stage=dest.with_name(dest.name+'.staging');stage.mkdir(parents=True,exist_ok=True)
def cp(src,rel):
 dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def js(rel,data):
 p=stage/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2,ensure_ascii=False))
cp(run/'selected/canonical.ply','asset.ply');cp(run/'selected/canonical.json','transform.json')
for f in ['load_validation.json','support_validation.json']:cp(run/'selected'/f,f)
cp(run/'selection_review.json','selection_review.json');js('assembly.json',r['assembly'])
packet=json.loads((W/'packets'/name/'segmentation.json').read_text());obs=packet['observations'][0]
for field,rel in [('image_path','source/original.png'),('anchor_path','source/anchor.png'),('rgba_path','source/anchor_rgba.png')]:cp(obs[field],rel)
fs=r['stages']['flux']['images'][0];cp(fs['path'],'source/flux.png');cp(Path(fs['path']).with_name('flux_rgba.png'),'source/flux_rgba.png')
source=json.loads((run/'scenes'/r['scene_id']/'logs/export_items_full.json').read_text())[0]
js('provenance.json',{'source':source['source'],'observations':packet,'reference_selection':source['real_view_selection'],'method':sel['method'],'flux_sha256':fs['sha256'],'asset_sha256':meta['asset_sha256'],'process_archive':str(run.relative_to(root)),'external_paths_are_audit_only':True})
for f in ['evidence.jpg','heading.jpg','turntable.mp4','render.json']:cp(run/'selected/render'/f,'previews/'+f)
for f in ['meta.json','visibility_diagnostics.json','contact_trajectory.json','checkpoint_load_audit.json','insert_splatad_6cam.mp4']:cp(ins/f,'insertion/'+mode+'/'+f)
trajectory_modes=list(dict.fromkeys(['static','straight','gentle_turn',mode]))
for m in trajectory_modes:cp(W/'configs'/('scene0103_'+m+'.npz'),'trajectories/'+m+'.npz')
prior=json.loads((W/'configs/metric_priors.json').read_text())[name];support=json.loads((run/'selected/support_validation.json').read_text());g=load_rigid(stage/'asset.ply');s=support['metric_scale'];v=g.xyz[0]*s
# Cover all centers plus conservative isotropic 3-sigma Gaussian support, not only robust quantiles.
radius=3*np.max(g.scale[0],axis=-1)*s;low=np.min(v-radius[:,None],axis=0);high=np.max(v+radius[:,None],axis=0)
js('placement.json',{'schema_version':1,'uid':'rider_'+name,'category':r['category'],'asset_kind':'rider_vehicle_rigid','ply':'asset.ply','fixed_pose':True,'forward':'+X','up':'+Z','units':'normalized','normalization_already_baked':True,'metric_scale_baked':False,'metric_scale':s,'target_length_m':prior['length_m'],'apply_scale_once':True,'dimensions_lwh_m':support['dimensions_lwh_m'],'scale_evidence':prior,'rotation_representation':'wxyz','opacity_representation':'logit','scale_representation':'log','collision':{'type':'oriented_box','center_xyz_m':((low+high)/2).tolist(),'size_lwh_m':(high-low).tolist(),'coverage':'all Gaussian centers plus 3 times largest local Gaussian sigma'},'trajectory_files':{m:'trajectories/'+m+'.npz' for m in trajectory_modes},'quality':'quality.json'})
js('quality.json',{'status':'candidate_pending_final_visual_acceptance','production_release':False,'numeric_load_passed':True,'short_trajectory_rendered':mode,'support':support,'limitations':['Fixed rider pose; wheels do not rotate','Fine face, hands, decals and unobserved parts synthesized','Source dimensional prior is not a calibrated generated scan','No fitted contact shadow; road-level appearance remains subject to review','Current insertion does not test deliberate foreground-object occlusion'],'requires':['final scene contact and occlusion review','catalog acceptance']})
js('package.json',{'schema_version':1,'asset':'asset.ply','placement':'placement.json','assembly':'assembly.json','provenance':'provenance.json','quality':'quality.json','preview':'previews/turntable.mp4','status':'candidate','files':sorted(str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file())})
os.rename(stage,dest);print(dest)
