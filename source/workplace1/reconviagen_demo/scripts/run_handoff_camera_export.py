from pathlib import Path
import os,subprocess,json,concurrent.futures
R=Path(__file__).resolve().parents[1];S=R/'scene_validation'
scenes=json.loads((S/'inputs/scenes.json').read_text());m=json.loads((R/'inputs/random10_04_white_suv/manifest.json').read_text())
def run(item):
 i,s=item;dest=S/'docs/camera_export'/s['scene'];dest.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(i),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCH_EXTENSIONS_DIR=str(R/'cache/torch_extensions_splatad'),MPLCONFIGDIR=str(R/'cache/matplotlib'),TORCH_HOME=str(R/'cache/torch_splatad'),PATH=str(R/'env_splatad/bin')+':/mnt/ojc/miniconda3/envs/splatad/bin:'+env['PATH'])
 cmd=[str(R/'env_splatad/bin/python'),str(R/'scripts/export_insertion_handoff_cameras.py'),'--config',s['config'],'--traj_npz',s['traj_npz'],'--asset_ply',str(R/'results/random10_04_white_suv/H_seed42/asset/canonical.ply'),'--target_len',str(m['metric_dimensions_lwh'][0]),'--reference_width',str(m['metric_dimensions_lwh'][1]),'--out_dir',str(dest),'--start_frame','30','--max_frames','60','--fps','12','--ground_mode','lower_road']
 with (dest/'export.log').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 print(s['scene'],p.returncode,flush=True);return p.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:codes=list(ex.map(run,enumerate(scenes)))
assert not any(codes),codes
