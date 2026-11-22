import subprocess,concurrent.futures,json,sys
from pathlib import Path
W=Path(__file__).resolve().parents[1]
items=[('niu_m1','19460','https://video.golem.de/wissenschaft/19460/niu-moto-in-aktion-herstellervideo.html')]
def one(item):
 name,vid,page=item;p=W/'sources'/name;p.mkdir(exist_ok=True);url=f'https://video.golem.de/download/{vid}?q=medium'
 (p/'source.json').write_text(json.dumps(dict(source_id='golem_'+vid,page=page,download_url=url,license='No explicit reuse license found. Internal research preview; no redistribution rights assumed.',selection_status='candidate_not_yet_reviewed'),indent=2))
 if not (p/'preview.mp4').exists():
  code=subprocess.run(['curl','-L','--fail','--retry','1','--max-time','240','--max-filesize','150000000','-o',str(p/'preview.mp4.part'),url],stdout=subprocess.DEVNULL,stderr=(p/'download.log').open('w')).returncode
  if code:print(name,'DOWNLOAD_FAILED',code,flush=True);return
  (p/'preview.mp4.part').rename(p/'preview.mp4')
 r=subprocess.run(['/mnt/ojc/miniconda3/envs/4danyone-sam2/bin/python',str(W/'src/video_preview.py'),str(p/'preview.mp4'),'2'],capture_output=True,text=True)
 print(name,r.returncode,r.stdout,r.stderr[-500:],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(one,items))
