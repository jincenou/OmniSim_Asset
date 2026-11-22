from pathlib import Path
import json,os,subprocess,sys,hashlib
import numpy as np
import pytest
TOOL='/mnt/ojc/workplace1/human_minimal_toolkit_20261105/compact_humans.py'

def make_source(root):
 root.mkdir();rows=[]
 for name,skip in [('H1',False),('H2',True)]:
  p=root/name;p.mkdir();times=np.array([0.,1.]);xyz=np.zeros((2,3,3),dtype=np.float32);xyz[:,:,2]=1;xyz[1,:,0]=.2
  q=np.zeros((2,3,4),dtype=np.float32);q[:,:,0]=1
  np.savez(p/'asset_4dgs.npz',frame_xyz=xyz,frame_rotation=q,frame_scaling=np.full((2,3,3),.1),opacity=np.ones((3,1))*.5,shs=np.ones((3,3))*.25,use_rgb=[1],frame_timestamps=times,canonical_unused=np.zeros((3,3)))
  np.savez(p/'placement_playback.npz',frame_timestamps=times,root_xy=np.zeros((2,2)),root_yaw=np.zeros(2),asset_to_standard=np.eye(4),ground_offset=np.array(0.))
  (p/'manifest.json').write_text(json.dumps({'fps':1,'frame_count':2,'asset':{'path':'asset_4dgs.npz'}}))
  (p/'insertion.json').write_text(json.dumps({'practical_playback':{'approximate_placement_available':True,'coordinate_convention':'right_handed_x_forward_y_left_z_up','action':{'label':'standing'},'native_speed_model_mps':0.,'arrays':'placement_playback.npz'}}))
  (p/'demo.mp4').write_bytes(b'preview preserved by rename')
  (p/'runtime').mkdir();(p/'runtime/model.pth').write_bytes(b'obsolete source model')
  outside=root.parent/'outside.txt';outside.write_bytes(b'keep outside source');(p/'linked_source').symlink_to(outside)
  rows.append(dict(global_uid=name,package_path=name,size_xyz=[1,1,2],skip_insertion=skip,insertion_exclusion_reason='test exclusion' if skip else None))
 (root/'drivelab_asset.json').write_text(json.dumps({'assets':rows}))

def run(root,reports,*extra):
 return subprocess.run([sys.executable,TOOL,'inplace','--source',str(root),'--report-dir',str(reports),'--jobs','1',*extra],text=True,capture_output=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'})

def test_requires_explicit_apply(tmp_path):
 root=tmp_path/'human';make_source(root);r=run(root,tmp_path/'reports');assert r.returncode!=0
 assert (root/'H1/asset_4dgs.npz').exists()

def test_partial_resume_and_preserved_preview(tmp_path):
 root=tmp_path/'human';make_source(root);reports=tmp_path/'reports'
 a=run(root,reports,'--apply','--asset-id','H1');assert a.returncode==0,a.stderr
 assert {p.name for p in (root/'H1').iterdir()}=={'asset.json','runtime.npz'}
 assert (root/'H2/manifest.json').exists() and not (root/'library.json').exists()
 assert json.loads((root/'drivelab_asset.json').read_text())['assets'][0]['runtime_metadata']=='H1/asset.json'
 assert (root/'previews/H1.mp4').read_bytes()==b'preview preserved by rename'
 assert (tmp_path/'outside.txt').read_bytes()==b'keep outside source'
 b=run(root,reports,'--apply','--resume');assert b.returncode==0,b.stderr
 index=json.loads((root/'library.json').read_text());assert index['assets']==[{'metadata':'H1/asset.json'}]
 assert index['excluded_assets'][0]['id']=='H2';assert not (root/'drivelab_asset.json').exists()
 for name in ['H1','H2']:assert {p.name for p in (root/name).iterdir()}=={'asset.json','runtime.npz'}
 before=hashlib.sha256((root/'H1/runtime.npz').read_bytes()).hexdigest()
 c=run(root,reports,'--apply','--resume');assert c.returncode==0,c.stderr
 assert hashlib.sha256((root/'H1/runtime.npz').read_bytes()).hexdigest()==before
 assert json.loads((reports/'summary.json').read_text())['complete']

def test_invalid_asset_is_not_deleted(tmp_path):
 root=tmp_path/'human';make_source(root)
 with np.load(root/'H1/asset_4dgs.npz') as z:arr={k:z[k] for k in z.files}
 arr['frame_rotation'][:]=0;np.savez(root/'H1/asset_4dgs.npz',**arr)
 r=run(root,tmp_path/'reports','--apply','--asset-id','H1');assert r.returncode!=0
 assert (root/'H1/manifest.json').exists();assert (root/'H1/runtime/model.pth').exists();assert not (root/'library.json').exists()
