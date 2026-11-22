"""Attach a finished validation only when asset and trajectory hashes match."""
from pathlib import Path
import sys,json,hashlib,shutil,argparse,subprocess
W=Path(__file__).resolve().parents[1];root=W.parents[2]
parser=argparse.ArgumentParser();parser.add_argument('name');parser.add_argument('mode');parser.add_argument('--recover-completed-render',action='store_true');args=parser.parse_args()
name,mode=args.name,args.mode;run=W/'runs'/name;ins=run/('insertion_'+mode)
exit_path=run/('insert_'+mode+'.exit');recovery=None
if exit_path.exists():
 if exit_path.read_text().strip()!='0':raise ValueError('Validation did not finish successfully')
elif not args.recover_completed_render:raise ValueError('Missing controller exit; explicit completed-artifact recovery required')
meta=json.loads((ins/'meta.json').read_text());sel=json.loads((run/'selected/selected.json').read_text());cat=sel['row']['category'];dest=root/'rigid'/cat/('rider_'+name if name.startswith('bicycle_nusc') else 'rider_web_'+name)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
traj=W/'configs'/('scene0103_'+mode+'.npz');assert sha(dest/'asset.ply')==meta['asset_sha256'];assert sha(traj)==meta['trajectory_sha256']
assert meta['n_frames']>=24
if not exit_path.exists():
 video=ins/'insert_splatad_6cam.mp4';bin_dir=Path('/mnt/ojc/miniconda3/envs/splatad/bin')
 probe=json.loads(subprocess.check_output(['/home/wmj/miniconda3/envs/sim/bin/ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(video)]))
 frames=int(probe['streams'][0]['nb_read_frames'])
 if frames!=meta['n_frames']:raise ValueError('Recovered video frame count differs from renderer metadata')
 subprocess.run([str(bin_dir/'ffmpeg'),'-v','error','-xerror','-i',str(video),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,timeout=120)
 recovery=dict(controller_exit='missing; not inferred',evidence='Completed renderer metadata, matching asset/trajectory hashes and successful full video decode',decoded_frames=frames,video_sha256=sha(video))
out=dest/'insertion'/mode;out.mkdir(exist_ok=False)
for f in ['meta.json','visibility_diagnostics.json','contact_trajectory.json','checkpoint_load_audit.json','insert_splatad_6cam.mp4']:shutil.copy2(ins/f,out/f)
if recovery:(out/'recovery.json').write_text(json.dumps(recovery,indent=2))
shutil.copy2(traj,dest/'trajectories'/(mode+'.npz'))
p=dest/'placement.json';cfg=json.loads(p.read_text());cfg.setdefault('trajectory_files',{})[mode]='trajectories/'+mode+'.npz';cfg.setdefault('insertion',{})[mode]='insertion/'+mode+'/insert_splatad_6cam.mp4';p.write_text(json.dumps(cfg,indent=2))
p=dest/'quality.json';q=json.loads(p.read_text());q.setdefault('additional_validations',{})[mode]={'render_completed':True,'asset_hash_verified':True,'trajectory_hash_verified':True,'visual_review':'pending'};p.write_text(json.dumps(q,indent=2))
p=dest/'package.json';pack=json.loads(p.read_text());pack['files']=sorted(str(f.relative_to(dest)) for f in dest.rglob('*') if f.is_file() and f!=p);p.write_text(json.dumps(pack,indent=2));print(dest,mode)
