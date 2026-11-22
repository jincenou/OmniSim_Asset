"""Package selected demos with portable local evidence, explicit scale and limitations."""
import sys,json,shutil,hashlib,subprocess
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'src/DriveLab'))
from asset_extraction.gaussians import load_rigid
from asset_extraction.export import inspect_ply
OUT=Path('/mnt/ojc/workplace1/asset_library/rigid/motorcycle')
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def copy(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
name=sys.argv[1];run=W/'runs'/name;
required_mode={'nuscenes_b2710614':'static','niu_blue':'straight','niu_black':'gentle_turn'}[name]
ins=run/('insertion_'+required_mode)
meta=json.loads((ins/'meta.json').read_text());assert meta['n_frames']>=12
diag=json.loads((ins/'visibility_diagnostics.json').read_text());front=[d for d in diag if d['sensor']=='FRONT'];assert len(front)==meta['n_frames'] and all(d['visible_pixels_01']>100 for d in front)
s=json.loads((run/'selected/selected.json').read_text());r=s['row'];method=s['method'];c=json.loads((run/'selected/canonical.json').read_text());dest=OUT/({'nuscenes_b2710614':'H_rider_nuscenes_b2710614','niu_blue':'B0_rider_niu_blue','niu_black':'H_rider_niu_black'}[name])
if dest.exists():raise FileExistsError(dest)
dest.mkdir(parents=True);copy(run/'selected/canonical.ply',dest/'asset.ply');copy(Path(r['raw_path']),dest/'raw.ply');copy(run/'selected/canonical.json',dest/'transform.json')
packet=json.loads((W/'packets'/name/'segmentation.json').read_text());source=packet['observations'][0]
copy(Path(source['image_path']),dest/'source/original.png');copy(Path(source['anchor_path']),dest/'source/anchor.png');copy(Path(source['rgba_path']),dest/'source/anchor_rgba.png')
write(dest/'source/observations.json',packet);write(dest/'assembly.json',r['assembly'])
for i,real in enumerate(r['real_images']):copy(Path(real['rgba_path']),dest/f'source/reference_{i}.png')
fs=r['stages']['flux']['images'][0];copy(Path(fs['path']),dest/'source/flux.png');copy(Path(fs['path']).with_name('flux_rgba.png'),dest/'source/flux_rgba.png')
render=run/'selected/render'
for f in ['turntable.mp4','evidence.jpg','heading.jpg','render.json']:copy(render/f,dest/f)
length=2.267 if name.startswith('nuscenes') else 1.949
basis={'source':'nuScenes native combined box','source_size_wlh':source['source_observation']['size_wlh'],'confidence':'medium'} if name.startswith('nuscenes') else {'source':'https://global.niu.com/en-us/product/mqi-gt','length_prior_m':1.949,'confidence':'medium','limitation':'Model-family dimension prior; accessory extension and generated proportions introduce uncertainty, not calibrated scan'}
g=load_rigid(dest/'asset.ply');v=g.xyz[0];lo,hi=np.quantile(v,[.005,.995],axis=0);scale=length/(hi[0]-lo[0]);dims=(hi-lo)*scale
support={}
for label,sel in [('front',v[:,0]>.2*(hi[0]-lo[0])),('rear',v[:,0]<-.2*(hi[0]-lo[0]))]:
 z=v[sel&(g.opacity>.5)&(np.abs(v[:,1])<.15*(hi[0]-lo[0])),2]
 support[label+'_wheel_lower_quantile_m']=float(np.quantile(z,.005)*scale)
support['wheel_support_difference_m']=abs(support['front_wheel_lower_quantile_m']-support['rear_wheel_lower_quantile_m']);support['method']='Central lateral strip at both longitudinal ends; 0.5% lower quantile, checked against wheel location in rendered views'
source_info=json.loads((run/'scenes'/r['scene_id']/'logs/export_items_full.json').read_text())[0]
write(dest/'provenance.json',dict(source=source_info['source'],reference_selection=source_info['real_view_selection'],generation=dict(method=method,seed=42,flux_sha256=fs['sha256'],ply_sha256=hashlib.sha256((dest/'asset.ply').read_bytes()).hexdigest(),stages=r['stages'])))
limitations=['Fixed pose: no independent wheel, handlebar or limb motion','Unobserved surfaces, facial details and small text synthesized; demo is not a faithful scan','Real-scene shading uses existing SplatAD color adaptation; no shadow fitting']
quality=dict(status='demo_reviewed',reviewer='Codex visual inspection of source, matte and 16-view reconstruction',assembly_structure_verified=True,heading_verified=c['heading_verified'],support=support,metric_scale_evidence=basis,limitations=limitations,ply_validation=inspect_ply(dest/'asset.ply'),delete_rejected=False)
quality['insertion_visibility']=dict(mode=required_mode,frames=len(front),min_visible_pixels=min(d['visible_pixels_01'] for d in front),edge_clipped_frames=sum(d['edge_clipped'] for d in front),checkpoint_load_audit=meta['checkpoint_load_audit'])
write(dest/'quality.json',quality)
copy(run/'selected/load_validation.json',dest/'load_validation.json')
copy(run/'selection_review.json',dest/'selection_review.json')
config=dict(schema_version=1,uid=name,category='motorcycle',asset_kind='rider_vehicle_rigid',ply='asset.ply',raw_ply='raw.ply',assembly='assembly.json',fixed_pose=True,forward='+X',up='+Z',units='normalized',normalization_already_baked=True,metric_scale_baked=False,metric_scale=float(scale),target_length_m=length,dimensions_lwh_m=dims.tolist(),scale_confidence=basis['confidence'],apply_scale_once=True,rotation_representation='wxyz',opacity_representation='logit',scale_representation='log',sh_degree=0,placement_origin='horizontal robust-bounds center / wheel-bottom vicinity',collision=dict(type='oriented_box',size_lwh_m=(dims+np.array([.1,.1,.05])).tolist()),quality='quality.json',provenance='provenance.json',turntable='turntable.mp4',insertion={})
for mode in ['static','straight','gentle_turn']:
 copy(W/'configs'/('scene0103_'+mode+'.npz'),dest/('trajectories/'+mode+'.npz'))
 ins=run/('insertion_'+mode)
 if (ins/'meta.json').exists():
  for f in ['meta.json','visibility_diagnostics.json','contact_trajectory.json','checkpoint_load_audit.json','insert_splatad_6cam.mp4']:copy(ins/f,dest/('insertion/'+mode+'/'+f))
  video=dest/('insertion/'+mode+'/front_camera.mp4')
  subprocess.run(['/mnt/ojc/miniconda3/envs/splatad/bin/ffmpeg','-hide_banner','-loglevel','error','-i',str(ins/'insert_splatad_6cam.mp4'),'-vf','crop=iw/3:ih/2:iw/3:0','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(video)],check=True)
  if (ins/'contact_sheet.jpg').exists():copy(ins/'contact_sheet.jpg',dest/('insertion/'+mode+'/contact_sheet.jpg'))
  config['insertion'][mode]='insertion/'+mode+'/front_camera.mp4'
write(dest/'placement.json',config);print(dest)
