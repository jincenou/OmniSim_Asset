import json,sys,subprocess
from pathlib import Path
W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(W/'DriveLab'))
from asset_extraction.export import render_evidence
d=W/'runs'/sys.argv[1]/'curation'
render_evidence(d/'canonical.ply',d/'final_render',size=768,save_frames=True)
render_evidence(d/'canonical.ply',d/'low_render',size=512,elevation_deg=0,save_frames=True)
subprocess.run(['/mnt/ojc/miniconda3/envs/qwen35_vlm/bin/ffmpeg','-y','-v','error','-framerate','4','-i',str(d/'final_render/view_%02d.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(d/'orbit.mp4')],check=True)
