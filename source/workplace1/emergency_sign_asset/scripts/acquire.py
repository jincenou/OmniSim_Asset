import concurrent.futures, datetime, hashlib, json, pathlib, subprocess, urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
FF='/mnt/ojc/miniconda3/envs/qwen35_vlm/bin/'
PY='/mnt/ojc/DriveLab/.venv/bin/python'
SOURCES=[
('V01','https://www.youtube.com/watch?v=iU46XTTL13U',None,'Deflecto'),
('V02','https://www.etrailer.com/tv-Demo-Erickson-Emergency-Supplies-Warning-Triangles-EM05310.aspx','https://www.etrailer.com/static/images/video/Demo-Erickson-Emergency-Supplies-Warning-Triangles-EM05310_web.mp4','etrailer'),
('V04','https://www.youtube.com/watch?v=aXuwYwRjuyc',None,'Traffic Safety Supply Company'),
('V06','https://www.youtube.com/watch?v=OMJ63cysPiU',None,'Traffic Safety Store'),
('V09','https://www.youtube.com/watch?v=f6rNA4y2pkE',None,'Blaulicht-Magazin'),
('V10','https://www.pexels.com/video/road-closed-sign-7297442/','https://videos.pexels.com/video-files/7297442/7297442-hd_1920_1080_30fps.mp4','Brayds Channel'),
]
def run(cmd,p,name,timeout=300):
 r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
 (p/(name+'.stdout')).write_bytes(r.stdout); (p/(name+'.stderr')).write_bytes(r.stderr)
 if r.returncode: raise RuntimeError(f'{name} exit {r.returncode}: '+r.stderr.decode(errors='replace')[-700:])
 return r.stdout

def acquire(s):
 sid,url,media,author=s;p=ROOT/'sources'/sid;p.mkdir(exist_ok=True)
 if (p/'acquisition.json').exists():
  old=json.loads((p/'acquisition.json').read_text())
  if old.get('status')=='downloaded_full_decode_ok': return old
  archive=p/('attempt_'+datetime.datetime.now().strftime('%H%M%S'));archive.mkdir(exist_ok=True)
  import shutil
  for prior in list(p.glob('*.json'))+list(p.glob('*.stdout'))+list(p.glob('*.stderr')): shutil.copy2(prior,archive/prior.name)
 row=dict(source_id=sid,page_url=url,media_url=media,author_supplied=author,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),license_status='not_verified',status='pending')
 def save(): (p/'acquisition.json').write_text(json.dumps(row,indent=2,ensure_ascii=False))
 save()
 try:
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=40) as r: (p/'page.html').write_bytes(r.read())
  except Exception as e: row['page_error']=str(e)
  dest=p/'source.mp4'
  if media:
   req=urllib.request.Request(media,method='HEAD',headers={'User-Agent':'Mozilla/5.0'})
   with urllib.request.urlopen(req,timeout=40) as r: row['pre_download_probe']=dict(status=r.status,headers=dict(r.headers),resolved_url=r.url)
   save()
   if not dest.exists():
    with urllib.request.urlopen(urllib.request.Request(media,headers={'User-Agent':'Mozilla/5.0'}),timeout=60) as r, (p/'source.part').open('wb') as f:
     while b:=r.read(1024*1024): f.write(b)
    (p/'source.part').rename(dest)
  else:
   raw=run([PY,'-m','yt_dlp','--no-playlist','--skip-download','--dump-single-json','--socket-timeout','25','--retries','1',url],p,'metadata',150)
   metadata=json.loads(raw);row['metadata']={k:metadata.get(k) for k in ['id','title','uploader','channel','upload_date','duration','license','webpage_url']};save()
   run([PY,'-m','yt_dlp','--no-playlist','--socket-timeout','30','--retries','1','--ffmpeg-location',FF,'-f','bestvideo[height<=1080][ext=mp4]/best[height<=1080]','--write-info-json','-o',str(dest),url],p,'download',300)
  row['sha256']=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest();row['bytes']=dest.stat().st_size
  probe=json.loads(run([FF+'ffprobe','-v','error','-show_format','-show_streams','-of','json',str(dest)],p,'ffprobe'))
  row['probe']=probe
  run([FF+'ffmpeg','-v','error','-i',str(dest),'-map','0:v:0','-f','null','-'],p,'decode',300)
  row['status']='downloaded_full_decode_ok'
 except Exception as e: row['status']='failed';row['error']=f'{type(e).__name__}: {e}'
 save(); print(sid,row['status'],row.get('error',''),flush=True);return row
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex: rows=list(ex.map(acquire,SOURCES))
 (ROOT/'sources'/'registry.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
